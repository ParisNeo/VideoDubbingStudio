import os
import json
import logging
import torch
import warnings
import traceback
from threading import Lock
from typing import Any, Tuple, Dict, Optional, List
from pathlib import Path
import numpy as np
import time

# Environment configuration to avoid large pre-reserved CUDA segments.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Configuration
from dotenv import load_dotenv

# Third-Party Libraries
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
    Wav2Vec2FeatureExtractor,
    WavLMForXVector,
)
from huggingface_hub import hf_hub_download
from lollms_client import LollmsClient

# F5-TTS Imports (Graceful handling if missing)
try:
    from f5_tts.model import DiT
    from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process
    F5_AVAILABLE = True
except ImportError:
    F5_AVAILABLE = False
    print("Warning: F5-TTS not installed. Voice cloning will not work.")

# Demucs for background separation
try:
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    DEMUX_AVAILABLE = True
except ImportError:
    DEMUX_AVAILABLE = False
    print("Warning: Demucs not installed. Background separation unavailable.")

# Setup Logging & Environment
load_dotenv()
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

logger = logging.getLogger("core.resources")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Paths
BASE_DIR = Path(__file__).parent.parent
STATUS_FILE = BASE_DIR / "task_history.json"
STATUS_BACKUP = STATUS_FILE.with_suffix(".bak")
STATUS_TEMP = STATUS_FILE.with_suffix(".tmp")


class ResourceManager:
    """Singleton GPU resource manager with 8GB-optimized loading."""
    
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ResourceManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use float16 for GPU to save memory, float32 for CPU
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.use_flash_attn = False  # Disable unless explicitly enabled (saves VRAM)

        # 8GB GPU optimization: limit CUDA memory fraction
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.85)  # Leave headroom

        # Model Cache with LRU-like behavior
        self._models: Dict[str, Any] = {}
        self._model_access_times: Dict[str, float] = {}
        self._max_cached_models = 3  # Limit concurrent loaded models

        # Configuration
        self.lollms_url = os.getenv("LOLLMS_URL", "http://localhost:9642")
        self.lollms_service_key = os.getenv("LOLLMS_API_KEY", "")
        self.lollms_model = os.getenv("LOLLMS_MODEL_NAME", "mistral")
        self.verify_ssl_certificate = bool(os.getenv("LOLLMS_VERIFY_SSL_CERTIFICATE", True))

        # Demucs model (lazy loaded)
        self._demucs_model = None

        logger.info(
            f"ResourceManager initialized: device={self.device}, "
            f"dtype={self.dtype}, torch={torch.__version__}"
        )
        self._initialized = True

    def _cleanup_if_needed(self):
        """Unload oldest models if we exceed cache limit."""
        if len(self._models) > self._max_cached_models:
            # Sort by last access time
            sorted_models = sorted(
                self._model_access_times.items(),
                key=lambda x: x[1]
            )
            # Unload oldest (keep most recent)
            to_unload = sorted_models[:-self._max_cached_models]
            for key, _ in to_unload:
                if key in self._models:
                    logger.info(f"Unloading model to save VRAM: {key}")
                    del self._models[key]
                    del self._model_access_times[key]
            torch.cuda.empty_cache()

    def _load_to_cache(self, key: str, loader_func):
        """Thread-safe lazy loader with LRU eviction."""
        if key not in self._models:
            with self._lock:
                if key not in self._models:
                    self._cleanup_if_needed()
                    logger.info(f"Loading model: {key}...")
                    try:
                        self._models[key] = loader_func()
                        # Simple timestamp tracking - no CUDA events needed
                        self._model_access_times[key] = time.time()
                        logger.info(f"Loaded {key}")
                    except Exception as e:
                        tb_str = traceback.format_exc()
                        logger.error(f"Failed to load {key} with full traceback:\n{tb_str}")
                        raise RuntimeError(f"Failed to load model {key}: {str(e)}\n\nFull traceback:\n{tb_str}")
        # Update access time
        self._model_access_times[key] = time.time()
        return self._models[key]

    # --------------------------------------------------------------------
    # 1. Whisper (Speech-to-Text) - Optimized for 8GB and Windows compatible
    # --------------------------------------------------------------------
    def get_whisper(self, model_name: Optional[str] = None):
        """
        Get Whisper model for speech-to-text.
        
        Args:
            model_name: Model to use (tiny, base, small, medium, large-v1, large-v2, large-v3)
                       Defaults to WHISPER_MODEL env var or 'large-v2'
        """
        def _loader():
            # Model selection with VRAM-aware defaults
            WHISPER_MODELS = {
                'tiny': {'id': 'openai/whisper-tiny', 'vram_mb': 400, 'speed': 'fastest', 'quality': 'lowest'},
                'base': {'id': 'openai/whisper-base', 'vram_mb': 800, 'speed': 'very fast', 'quality': 'low'},
                'small': {'id': 'openai/whisper-small', 'vram_mb': 2000, 'speed': 'fast', 'quality': 'medium'},
                'medium': {'id': 'openai/whisper-medium', 'vram_mb': 5000, 'speed': 'medium', 'quality': 'good'},
                'large-v1': {'id': 'openai/whisper-large', 'vram_mb': 10000, 'speed': 'slow', 'quality': 'excellent'},
                'large-v2': {'id': 'openai/whisper-large-v2', 'vram_mb': 10000, 'speed': 'slow', 'quality': 'excellent'},
                'large-v3': {'id': 'openai/whisper-large-v3', 'vram_mb': 10000, 'speed': 'slow', 'quality': 'best'},
            }
            
            # Get model from param, env var, or default
            import os
            selected = (model_name or 
                       os.getenv("WHISPER_MODEL", "large-v2")).lower().strip()
            
            # Map legacy names
            if selected == 'large':
                selected = 'large-v2'
            
            model_info = WHISPER_MODELS.get(selected, WHISPER_MODELS['large-v2'])
            model_id = model_info['id']
            
            logger.info(f"Loading Whisper model: {selected} ({model_id}) - "
                       f"VRAM: ~{model_info['vram_mb']}MB, Quality: {model_info['quality']}")

            try:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    attn_implementation="sdpa",  # More memory efficient than flash_attn_2
                ).to(self.device)

                processor = AutoProcessor.from_pretrained(model_id)

                # Very conservative batch size for 8GB GPU
                batch_size = 1 if self.device == "cuda" else 4
                
                # CRITICAL: Use return_tensors="pt" and chunk_length_s for Windows compatibility
                # This ensures the pipeline processes audio correctly on all platforms
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    torch_dtype=self.dtype,
                    device=self.device,
                    batch_size=batch_size,
                    chunk_length_s=30,  # Process in 30-second chunks
                    return_timestamps=False,  # Don't return timestamps to simplify output format
                )
                
                # Monkey-patch to handle Windows path issues
                original_call = pipe.__call__
                
                def windows_safe_call(inputs, **kwargs):
                    """
                    Wrapper to handle Windows-specific issues with audio inputs.
                    Ensures file paths are properly resolved and audio arrays are handled correctly.
                    """
                    import soundfile as sf
                    import numpy as np
                    from pathlib import Path
                    
                    # If inputs is a string (file path), ensure it's absolute
                    if isinstance(inputs, str):
                        path = Path(inputs)
                        if not path.is_absolute():
                            inputs = str(path.absolute())
                    
                    # If inputs is a Path object, convert to absolute string
                    elif isinstance(inputs, Path):
                        inputs = str(inputs.absolute())
                    
                    # If it's a numpy array, we need to handle it specially on Windows
                    # The pipeline should handle this, but Windows can be finicky
                    elif isinstance(inputs, np.ndarray):
                        # Don't try to convert arrays - let the pipeline handle it
                        # But ensure it's the right shape (mono)
                        if inputs.ndim > 1:
                            inputs = inputs.mean(axis=1)
                    
                    # Call the original pipeline with processed inputs
                    return original_call(inputs, **kwargs)
                
                # Replace the call method
                pipe.__call__ = windows_safe_call
                
                return pipe
                
            except Exception as e:
                tb_str = traceback.format_exc()
                logger.error(f"Whisper model loading failed with traceback:\n{tb_str}")
                raise RuntimeError(f"Failed to load Whisper: {str(e)}\n\nFull traceback:\n{tb_str}")

        return self._load_to_cache("whisper", _loader)

    # --------------------------------------------------------------------
    # 2. Demucs (Background Separation)
    # --------------------------------------------------------------------
    def get_demucs(self):
        """Get or load Demucs model for background/foreground separation."""
        if not DEMUX_AVAILABLE:
            raise ImportError("Demucs not available. Install: pip install demucs")
        
        if self._demucs_model is None:
            try:
                logger.info("Loading Demucs model...")
                self._demucs_model = get_model("htdemucs")
                if self.device == "cuda":
                    self._demucs_model = self._demucs_model.to(self.device)
            except Exception as e:
                tb_str = traceback.format_exc()
                logger.error(f"Demucs loading failed with traceback:\n{tb_str}")
                raise RuntimeError(f"Failed to load Demucs: {str(e)}\n\nFull traceback:\n{tb_str}")
        
        return self._demucs_model

    # --------------------------------------------------------------------
    # 3. Speaker Encoder (WavLM for Diarization)
    # --------------------------------------------------------------------
    def get_speaker_encoder(self) -> Tuple[Any, Any]:
        def _loader():
            try:
                model_id = "microsoft/wavlm-base-plus-sv"
                feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
                model = WavLMForXVector.from_pretrained(model_id).to(self.device)
                # Enable gradient checkpointing for memory savings if needed
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
                return feature_extractor, model
            except Exception as e:
                tb_str = traceback.format_exc()
                logger.error(f"Speaker encoder loading failed with traceback:\n{tb_str}")
                raise RuntimeError(f"Failed to load speaker encoder: {str(e)}\n\nFull traceback:\n{tb_str}")

        return self._load_to_cache("speaker_encoder", _loader)

    # --------------------------------------------------------------------
    # 4. F5-TTS (Voice Cloning) - 8GB Optimized
    # --------------------------------------------------------------------
    def get_f5_tts(self) -> Tuple[Any, Any, Any]:
        """Returns model, vocoder, and voice cloning function."""
        if not F5_AVAILABLE:
            raise ImportError("F5-TTS library is not installed.")

        def _loader():
            repo_id = "SWivid/F5-TTS"
            filename = "F5TTS_Base/model_1200000.safetensors"  # Use safetensors if available
            
            logger.info(f"Fetching F5-TTS checkpoint...")
            try:
                try:
                    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
                except:
                    # Fallback to .pt if safetensors not available
                    filename_pt = "F5TTS_Base/model_1200000.pt"
                    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename_pt)

                model_cfg = dict(
                    dim=1024,
                    depth=22,
                    heads=16,
                    ff_mult=2,
                    text_dim=512,
                    conv_layers=4,
                )

                # Load with memory-efficient settings
                model = load_model(
                    DiT,
                    model_cfg,
                    ckpt_path,
                    mel_spec_type="vocos",
                    device=self.device,
                )
                
                # Enable eval mode and disable gradients
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False

                vocoder = load_vocoder(is_local=False)
                
                return model, vocoder
                
            except Exception as e:
                tb_str = traceback.format_exc()
                logger.error(f"F5-TTS loading failed with traceback:\n{tb_str}")
                raise RuntimeError(f"Failed to load F5-TTS: {str(e)}\n\nFull traceback:\n{tb_str}")

        return self._load_to_cache("f5_tts", _loader)

    # --------------------------------------------------------------------
    # 5. Lollms Client (Translation)
    # --------------------------------------------------------------------
    def get_lollms_client(self) -> Optional[LollmsClient]:
        """Returns a configured LollmsClient or None if connection fails."""
        def _loader():
            try:
                client = LollmsClient(
                    llm_binding_name="lollms",
                    llm_binding_config={
                        "host_address": self.lollms_url,
                        "service_key": self.lollms_service_key,
                        "model_name": self.lollms_model,
                        "verify_ssl_certificate": self.verify_ssl_certificate,
                    },
                )
                #  Test connection
                _ = client.generate_text("Hi", n_predict=1)
                return client
            except Exception as e:
                tb_str = traceback.format_exc()
                logger.warning(f"Could not initialise LollmsClient with traceback:\n{tb_str}")
                return None

        return self._load_to_cache("lollms_client", _loader)

    # --------------------------------------------------------------------
    # Utilities
    # --------------------------------------------------------------------
    def clear_cache(self, keep: List[str] = None):
        """Unload models to free VRAM, optionally keeping specific ones."""
        keep = keep or []
        with self._lock:
            to_remove = [k for k in list(self._models.keys()) if k not in keep]
            for k in to_remove:
                del self._models[k]
                if k in self._model_access_times:
                    del self._model_access_times[k]
            torch.cuda.empty_cache()
            logger.info(f"Cleared models: {to_remove}, kept: {keep}")

    def offload_to_cpu(self, model_key: str):
        """Move a specific model to CPU temporarily."""
        if model_key in self._models and torch.cuda.is_available():
            self._models[model_key] = self._models[model_key].cpu()
            torch.cuda.empty_cache()
            logger.info(f"Offloaded {model_key} to CPU")


# Singleton Instance
manager = ResourceManager()
