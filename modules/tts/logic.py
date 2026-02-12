"""
TTS generation logic with Intel SVML error handling and automatic fallback.
"""

import os
import sys
import gc
import time
import tempfile
import warnings
from typing import Tuple, Optional
import numpy as np
import soundfile as sf
import librosa

# Re-apply SVML workaround before any torch operations in this module
os.environ["MKL_ENABLE_INSTRUCTIONS"] = "SSE4_2"
os.environ["NPY_DISABLE_CPU_FEATURES"] = "AVX512F,AVX2,AVX"


def _is_svml_error(error_msg: str) -> bool:
    """Detect if error is related to Intel SVML/CPU instruction issues."""
    svml_keywords = [
        'svml', '__svml', 'llvm', 'intel', 'mkl', 
        'illegal instruction', 'symbol not found',
        'cpu dispatcher', 'avx', 'cosf8', 'sinf8', 'expf8'
    ]
    return any(kw in error_msg.lower() for kw in svml_keywords)


def generate_speech_f5_robust(
    text: str,
    ref_audio_path: str,
    model_obj,
    vocoder,
    device: str = "cuda",
    max_retries: int = 3
) -> Tuple[np.ndarray, int]:
    """
    F5-TTS inference with comprehensive SVML error handling.
    
    Automatically retries with CPU fallback when Intel vectorization errors occur.
    """
    from f5_tts.infer.utils_infer import infer_process
    
    # Prepare reference audio (10 second limit)
    ref_audio, sr = librosa.load(ref_audio_path, sr=24000, mono=True)
    ref_audio = ref_audio[:240_000]  # 10s @ 24kHz
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, ref_audio, 24000)
        ref_tmp = f.name
    
    last_error = None
    original_device = next(model_obj.parameters()).device
    
    for attempt in range(max_retries):
        try:
            # Re-apply environment settings before each attempt (defensive)
            os.environ["MKL_ENABLE_INSTRUCTIONS"] = "SSE4_2"
            os.environ["NPY_DISABLE_CPU_FEATURES"] = "AVX512F,AVX2,AVX"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OMP_NUM_THREADS"] = "1"
            
            # Force CPU on retry or if explicitly requested
            use_cpu = (attempt > 0) or (device == "cpu") or (os.name == 'nt')
            actual_device = "cpu" if use_cpu else device
            
            if use_cpu:
                torch.backends.mkldnn.enabled = False
                torch.backends.cudnn.enabled = False
                torch.set_num_threads(1)
                model = model_obj.cpu()
                voc = vocoder.cpu() if hasattr(vocoder, 'cpu') else vocoder
            else:
                model = model_obj
                voc = vocoder
            
            print(f"[F5-TTS] Attempt {attempt + 1}/{max_retries} on {actual_device.upper()}")
            
            # Run inference with all warnings suppressed
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gen_audio, sr_out, _ = infer_process(
                    ref_audio=ref_tmp,
                    ref_text="",
                    gen_text=text,
                    model_obj=model,
                    vocoder=voc,
                    mel_spec_type="vocos",
                    device=actual_device,
                )
            
            # Success: restore model to original device if moved
            if use_cpu and str(original_device) != 'cpu':
                try:
                    model_obj.to(original_device)
                    if hasattr(vocoder, 'to'):
                        vocoder.to(original_device)
                except Exception as e:
                    print(f"[F5-TTS] Warning: Could not restore models to {original_device}: {e}")
            
            # Process output audio
            if hasattr(gen_audio, 'cpu'):
                gen_audio = gen_audio.cpu().numpy()
            audio = np.asarray(gen_audio).flatten()
            
            # Normalize to prevent clipping
            peak = np.max(np.abs(audio))
            if peak > 0.95:
                audio = audio / peak * 0.95
            
            # Cleanup
            try:
                os.remove(ref_tmp)
            except:
                pass
            
            return audio, sr_out
            
        except Exception as e:
            last_error = e
            error_msg = str(e)
            
            if _is_svml_error(error_msg) and attempt < max_retries - 1:
                print(f"  ⚠ SVML/CPU error detected, retrying with CPU...")
                
                # Aggressive cleanup between attempts
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                time.sleep(0.5 * (attempt + 1))
                continue
            else:
                break
    
    # All retries failed: cleanup and raise
    try:
        os.remove(ref_tmp)
    except:
        pass
    
    raise RuntimeError(
        f"F5-TTS failed after {max_retries} attempts.\n"
        f"Last error: {last_error}\n\n"
        f"This is an Intel CPU/SVML compatibility issue. Solutions:\n"
        f"1. Use engine='fishspeech' (no local dependencies)\n"
        f"2. Install conda-forge PyTorch: conda install pytorch -c conda-forge\n"
        f"3. Set: SET NPY_DISABLE_CPU_FEATURES=AVX2 && python server.py"
    )


def generate_speech_fishspeech(
    text: str,
    ref_audio_path: str,
    api_url: Optional[str] = None
) -> Tuple[np.ndarray, int]:
    """
    FishSpeech API-based TTS (no local SVML/MKL dependencies).
    """
    import requests
    import base64
    import io
    
    url = api_url or os.getenv("FISH_SPEECH_API_URL", "http://127.0.0.1:8080/v1/tts")
    
    # Load and encode reference
    ref_audio, sr = sf.read(ref_audio_path)
    
    bio = io.BytesIO()
    sf.write(bio, ref_audio.astype(np.float32), sr, format='WAV')
    audio_b64 = base64.b64encode(bio.getvalue()).decode()
    
    payload = {
        "text": text,
        "reference_audio": audio_b64,
        "reference_text": "",
    }
    
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    
    result = response.json()
    audio_data = result.get('audio') or result.get('wav') or result.get('data')
    audio_bytes = base64.b64decode(audio_data)
    
    bio = io.BytesIO(audio_bytes)
    audio, out_sr = sf.read(bio)
    
    # Standardize to 24kHz
    if out_sr != 24000:
        audio = librosa.resample(audio, orig_sr=out_sr, target_sr=24000)
    
    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0.95:
        audio = audio / peak * 0.95
    
    return audio, 24000


def generate_speech(
    text: str,
    ref_audio_path: str,
    engine: Optional[str] = None,
    device: str = "cuda",
    **kwargs
) -> Tuple[np.ndarray, int]:
    """
    Unified speech generation with automatic engine selection and SVML protection.
    
    Args:
        text: Text to synthesize
        ref_audio_path: Path to reference audio file
        engine: 'f5', 'fishspeech', or None for platform-aware default
        device: Preferred compute device (ignored for FishSpeech)
    
    Returns:
        (audio_array, sample_rate)
    """
    # Auto-select engine if not specified
    if engine is None:
        engine = 'fishspeech' if os.name == 'nt' else 'f5'
    
    engine = engine.lower()
    
    if engine == 'fishspeech':
        return generate_speech_fishspeech(text, ref_audio_path, **kwargs)
    
    elif engine == 'f5':
        from core.resources import manager
        
        # Apply SVML protection
        os.environ["MKL_ENABLE_INSTRUCTIONS"] = "SSE4_2"
        os.environ["NPY_DISABLE_CPU_FEATURES"] = "AVX512F,AVX2,AVX"
        
        try:
            f5_model, vocoder = manager.get_f5_tts()
            return generate_speech_f5_robust(
                text=text,
                ref_audio_path=ref_audio_path,
                model_obj=f5_model,
                vocoder=vocoder,
                device=device
            )
        except RuntimeError as e:
            # Auto-fallback to FishSpeech on SVML errors
            if _is_svml_error(str(e)):
                print(f"[TTS] F5-TTS SVML error, auto-falling back to FishSpeech...")
                return generate_speech_fishspeech(text, ref_audio_path, **kwargs)
            raise
    
    else:
        raise ValueError(f"Unknown TTS engine: {engine}. Use 'f5' or 'fishspeech'.")


def get_default_tts_engine() -> str:
    """Return safe default TTS engine. Windows uses FishSpeech to avoid SVML."""
    if sys.platform == 'win32' or os.name == 'nt':
        return 'fishspeech'
    
    # Linux/Mac: use F5-TTS if GPU available
    try:
        if torch.cuda.is_available():
            return 'f5'
    except:
        pass
    
    return 'fishspeech'
