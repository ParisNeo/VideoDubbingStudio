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
from pathlib import Path  # ADDED: Missing import
import numpy as np
import soundfile as sf
import librosa
import pipmaster as pm
import logging
logger = logging.getLogger("logic")

# CRITICAL: Re-apply SVML workaround BEFORE any torch import or operation
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["MKL_ENABLE_INSTRUCTIONS"] = "SSE4_2"  # Disable AVX/AVX2/SVML
os.environ["NPY_DISABLE_CPU_FEATURES"] = "AVX512F,AVX2,AVX"  # Disable NumPy AVX

# Only import torch after environment is set
try:
    import torch
    # Disable MKL-DNN and limit threads
    torch.backends.mkldnn.enabled = False
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
except ImportError:
    pass


def _is_svml_error(error_msg: str) -> bool:
    """Detect if error is related to Intel SVML/CPU instruction issues."""
    svml_keywords = [
        'svml', '__svml', 'llvm', 'intel', 'mkl', 
        'illegal instruction', 'symbol not found',
        'cpu dispatcher', 'avx', 'cosf8', 'sinf8', 'expf8'
    ]
    return any(kw in error_msg.lower() for kw in svml_keywords)


def _clean_text_for_f5(text: str) -> str:
    """Normalize text to prevent F5-TTS internal sequence errors."""
    import re
    if not text: return ""
    # 1. Replace ellipses and multiple dashes with a single comma (F5 prefers commas for pauses)
    text = re.sub(r'\.{2,}', ',', text)
    text = re.sub(r'-{2,}', ',', text)
    # 2. Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # 3. Remove non-standard symbols but keep basic punctuation
    text = re.sub(r'[^\w\s,.\?!\']', '', text)
    # 4. Ensure it doesn't end with a hanging space
    text = text.strip()
    # 5. F5-TTS stability: ensure it ends with punctuation
    if text and text[-1] not in ('.', '!', '?', ','):
        text += '.'
    return text

def generate_speech_f5_robust(
    text: str,
    ref_audio_path: str,
    model_obj,
    vocoder,
    device: str = "cuda",
    max_retries: int = 3
) -> Tuple[np.ndarray, int]:
    """
    F5-TTS inference with comprehensive error handling.
    
    CRITICAL: F5-TTS requires PERFECT alignment between ref_text and ref_audio.
    Mismatches cause gibberish output.
    """
    from f5_tts.infer.utils_infer import infer_process
    
    # Pre-process text for alignment stability
    processed_text = _clean_text_for_f5(text)
    if not processed_text:
        logger.warning("F5-TTS: Empty text after cleaning, returning silence")
        return np.zeros(24000, dtype=np.float32), 24000
    
    logger.info(f"F5-TTS Input (cleaned): '{processed_text[:100]}...'")
    
    # CRITICAL: Prepare HIGH-QUALITY reference audio
    # F5 is extremely sensitive to noise/quality
    ref_audio, sr = librosa.load(ref_audio_path, sr=24000, mono=True)
    
    # Normalize reference audio to prevent clipping artifacts
    ref_audio = ref_audio.astype(np.float32)
    peak = np.max(np.abs(ref_audio))
    if peak > 0:
        ref_audio = ref_audio / peak * 0.95
    
    # CRITICAL: Use 3-8 second reference (F5 sweet spot)
    # Too short = poor quality, too long = alignment drift
    ref_duration = len(ref_audio) / 24000
    if ref_duration < 3.0:
        logger.warning(f"F5-TTS: Reference audio too short ({ref_duration:.1f}s), quality may suffer")
    elif ref_duration > 15.0:
        # Take middle 8 seconds to avoid silence at edges
        start_sample = int(len(ref_audio) * 0.2)  # Skip first 20%
        ref_audio = ref_audio[start_sample:start_sample + (8 * 24000)]
        logger.info(f"F5-TTS: Trimmed reference to 8s from {ref_duration:.1f}s")
    elif ref_duration > 8.0:
        # Take first 8 seconds
        ref_audio = ref_audio[:8 * 24000]
    
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
            
            logger.info(f"[F5-TTS] Attempt {attempt + 1}/{max_retries} on {actual_device.upper()}")
            
            # CRITICAL: Provide reference text for alignment
            # F5 uses this to align the reference audio with phonemes
            # Leaving it empty causes gibberish - we need APPROXIMATE transcript
            # of the reference audio, OR use the target text if reference is from
            # the same speaker saying similar content
            
            # Strategy: Use the target text as ref_text hint
            # This works when reference sample is from the same speaker
            # saying semantically similar content
            ref_text_hint = processed_text[:100]  # First 100 chars as alignment hint
            
            # Run inference with all warnings suppressed
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gen_audio, sr_out, _ = infer_process(
                    ref_audio=ref_tmp,
                    ref_text=ref_text_hint,  # CRITICAL: Provide alignment hint
                    gen_text=processed_text,
                    model_obj=model,
                    vocoder=voc,
                    mel_spec_type="vocos",
                    device=actual_device,
                    speed=1.0,  # Prevent speed artifacts
                )
            
            # Success: restore model to original device if moved
            if use_cpu and str(original_device) != 'cpu':
                try:
                    model_obj.to(original_device)
                    if hasattr(vocoder, 'to'):
                        vocoder.to(original_device)
                except Exception as e:
                    logger.warning(f"[F5-TTS] Could not restore models to {original_device}: {e}")
            
            # Process output audio
            if hasattr(gen_audio, 'cpu'):
                gen_audio = gen_audio.cpu().numpy()
            audio = np.asarray(gen_audio).flatten()
            
            # QUALITY CHECK: Detect gibberish audio
            # Gibberish typically has very low variance or extreme spikes
            audio_std = np.std(audio)
            audio_peak = np.max(np.abs(audio))
            
            if audio_std < 0.001:
                raise ValueError(f"F5-TTS produced near-silent audio (std={audio_std:.6f}), likely gibberish")
            
            if audio_peak > 10.0:
                raise ValueError(f"F5-TTS produced clipped audio (peak={audio_peak:.2f}), likely gibberish")
            
            # Check for NaN/Inf
            if not np.isfinite(audio).all():
                raise ValueError("F5-TTS produced NaN/Inf values, likely model error")
            
            logger.info(f"[F5-TTS] Generated {len(audio)/24000:.1f}s audio, std={audio_std:.4f}, peak={audio_peak:.4f}")
            
            # Normalize to prevent clipping
            if audio_peak > 0.95:
                audio = audio / audio_peak * 0.95
            elif audio_peak < 0.1 and audio_peak > 0:
                # Boost very quiet audio
                audio = audio / audio_peak * 0.5
            
            # Cleanup
            try:
                os.remove(ref_tmp)
            except:
                pass
            
            return audio, sr_out
            
        except Exception as e:
            last_error = e
            error_msg = str(e)
            
            logger.error(f"[F5-TTS] Attempt {attempt + 1} failed: {error_msg}")
            
            # Retry logic
            if attempt < max_retries - 1:
                if _is_svml_error(error_msg):
                    logger.warning("SVML/CPU error detected, retrying with CPU...")
                elif "gibberish" in error_msg.lower() or "near-silent" in error_msg.lower():
                    logger.warning("Quality check failed, retrying with adjusted parameters...")
                else:
                    logger.warning(f"Generic error, retrying...")
                
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


def generate_speech_lollms(
    text: str,
    voice: Optional[str] = None,
    model: Optional[str] = None,
    response_format: str = "wav",
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    audio_sample_path: Optional[str] = None,
    language: Optional[str] = None,
    max_retries: int = 2,
) -> Tuple[np.ndarray, int]:
    """
    LoLLMs TTS API-based speech generation using native /lollms/v1/audio/speech endpoint.
    
    Uses the LoLLMs native TTS endpoint with proper voice cloning support via audio_sample.
    
    Args:
        text: Text to synthesize
        voice: Voice name (optional, only used if no audio_sample_path)
        model: Model name (tts-1, tts-1-hd)
        response_format: Audio format (wav, mp3, opus, aac, flac, pcm)
        api_url: LoLLMs API URL (default: http://localhost:9642)
        api_key: LoLLMs API key
        audio_sample_path: Path to reference audio for voice cloning (primary voice source)
        language: Language code for TTS (e.g., 'en', 'es', 'fr')
        max_retries: Number of retries on failure
    
    Returns:
        (audio_array, sample_rate)
    """
    import requests
    import io
    import tempfile
    import subprocess
    import base64
    import time

    # CRITICAL FIX: Prevent XTTS from crashing on empty or symbol-only text
    import re
    # Check for at least one alphanumeric character
    if not text or not re.search(r'[a-zA-Z0-9]', str(text)):
        print(f"[LoLLMs TTS] Text '{text}' is empty or invalid for engine. Returning silence.")
        return np.zeros(24000, dtype=np.float32), 24000

    # Get LoLLMs configuration - ONLY use native LoLLMs endpoint
    base_url = api_url or os.getenv("LOLLMS_URL", "http://localhost:9642")
    key = api_key or os.getenv("LOLLMS_API_KEY", "")
    
    # ONLY use the LoLLMs native endpoint (OpenAI compatible doesn't work)
    url = base_url.rstrip("/") + "/lollms/v1/audio/speech"
    
    headers = {
        "Content-Type": "application/json",
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"
    
    # Build LoLLMs native payload
    # Priority: audio_sample (base64) > voice > default
    payload = {
            "input": text,
            "text": text,  # Fallback: Coqui XTTS backend sometimes strictly expects 'text' instead of 'input'
            "response_format": response_format,
            "speed": 1.0,
            "language": language
        }
    
    # Add model if specified
    if model:
        payload["model"] = model
    
    # Add language if specified
    if language:
        payload["language"] = language
    
    # Handle voice cloning via audio_sample (primary method)
    if audio_sample_path and Path(audio_sample_path).exists():
        try:
            with open(audio_sample_path, "rb") as f:
                ref_bytes = f.read()
                # Encode as base64 for audio_sample field
                payload["audio_sample"] = base64.b64encode(ref_bytes).decode('utf-8')
            print(f"[LoLLMs TTS] Using voice sample from {audio_sample_path}")
        except Exception as e:
            print(f"[LoLLMs TTS] Warning: Could not load reference audio: {e}")
            # Fall back to voice parameter if sample fails
            if voice:
                payload["voice"] = voice
    elif voice:
        # Use voice name if no sample provided
        payload["voice"] = voice
    else:
        # Default voice if nothing provided
        payload["voice"] = "alloy"
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            print(f"[LoLLMs TTS] Attempt {attempt + 1}/{max_retries} to {url}")
            
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            
            # Log error details for debugging
            if response.status_code != 200:
                error_detail = ""
                try:
                    error_detail = response.json()
                except:
                    error_detail = response.text[:500]
                print(f"[LoLLMs TTS] Error {response.status_code}: {error_detail}")
                last_error = f"HTTP {response.status_code}: {error_detail}"
                
                # Exponential backoff before retry
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[LoLLMs TTS] Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                continue
            
            # Success - process audio
            audio_bytes = response.content
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=f".{response_format}", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
            try:
                # Convert to WAV if needed
                if response_format in ["mp3", "opus", "aac", "flac"]:
                    wav_path = tmp_path.replace(f".{response_format}", ".wav")
                    try:
                        subprocess.run([
                            "ffmpeg", "-y", "-i", tmp_path,
                            "-ar", "24000", "-ac", "1",
                            wav_path
                        ], check=True, capture_output=True, timeout=30)
                        audio, out_sr = sf.read(wav_path)
                        try:
                            os.unlink(wav_path)
                        except:
                            pass
                    except Exception as conv_err:
                        print(f"[LoLLMs TTS] FFmpeg conversion failed: {conv_err}, trying direct load")
                        audio, out_sr = sf.read(tmp_path)
                else:
                    # WAV/FLAC/PCM - load directly
                    audio, out_sr = sf.read(tmp_path)
                
                # Ensure mono
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                
                # Resample to 24kHz
                if out_sr != 24000:
                    audio = librosa.resample(audio, orig_sr=out_sr, target_sr=24000)
                
                # Normalize
                peak = np.max(np.abs(audio))
                if peak > 0.95:
                    audio = audio / peak * 0.95
                elif peak < 0.01 and peak > 0:
                    audio = audio * 0.5 / peak
                
                return audio, 24000
                
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                
        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection failed to {url}: {e}"
            print(f"[LoLLMs TTS] {last_error}")
        except requests.exceptions.Timeout as e:
            last_error = f"Timeout connecting to {url}: {e}"
            print(f"[LoLLMs TTS] {last_error}")
        except Exception as e:
            last_error = f"Error with {url}: {e}"
            print(f"[LoLLMs TTS] {last_error}")
        
        # Exponential backoff before retry
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            print(f"[LoLLMs TTS] Retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    # All attempts failed
    raise RuntimeError(
        f"LoLLMs TTS failed after {max_retries} attempts. "
        f"Last error: {last_error}. "
        f"Please check: 1) LoLLMs is running at {base_url}, "
        f"2) TTS service is enabled in LoLLms settings, "
        f"3) The endpoint {url} is accessible, "
        f"4) Try using 'f5' or 'fishspeech' engine instead."
    )


def generate_speech(
    text: str,
    ref_audio_path: Optional[str] = None,
    engine: Optional[str] = None,
    device: str = "cuda",
    **kwargs
) -> Tuple[np.ndarray, int]:
    """
    Unified speech generation with automatic engine selection and SVML protection.
    
    Supported Engines:
        - xtts: Coqui XTTS v2 (recommended, most stable)
        - f5: F5-TTS (high quality, sensitive to input)
        - fishspeech: FishSpeech API (remote/local)
        - lollms: LoLLMs TTS API (remote)
        - bark: Suno Bark (creative, emotional)
        - styletts2: StyleTTS2 (fast, high quality)
        - piper: Piper TTS (fastest, no cloning)
    
    Args:
        text: Text to synthesize
        ref_audio_path: Path to reference audio file (required for cloning engines)
        engine: Engine name or None for auto-select
        device: Preferred compute device ('cuda' or 'cpu')
    
    Returns:
        (audio_array, sample_rate)
    """
    # Auto-select engine if not specified
    if engine is None:
        engine = get_default_tts_engine()
    
    engine = engine.lower()
    
    # Route to appropriate engine
    if engine == 'xtts':
        if not ref_audio_path:
            raise ValueError("XTTS requires a reference audio file for voice cloning")
        return generate_speech_xtts(text, ref_audio_path, device, **kwargs)
    
    elif engine == 'bark':
        return generate_speech_bark(text, device, **kwargs)
    
    elif engine == 'styletts2':
        if not ref_audio_path:
            raise ValueError("StyleTTS2 requires a reference audio file")
        return generate_speech_styletts2(text, ref_audio_path, device, **kwargs)
    
    elif engine == 'piper':
        voice = kwargs.get('voice', 'en_US-lessac-medium')
        return generate_speech_piper(text, voice, **kwargs)
    
    elif engine == 'fishspeech':
        if not ref_audio_path:
            raise ValueError("FishSpeech requires a reference audio file")
        return generate_speech_fishspeech(text, ref_audio_path, **kwargs)
    
    elif engine == 'lollms':
        # LoLLMs TTS now supports voice cloning via audio_sample
        # ref_audio_path is used for voice cloning if provided
        voice = kwargs.get('voice', 'alloy')
        model = kwargs.get('model', 'tts-1')
        response_format = kwargs.get('response_format', 'wav')  # Use WAV for better quality
        api_url = kwargs.get('api_url')
        api_key = kwargs.get('api_key')
        language = kwargs.get('language')  # Language parameter for TTS
        return generate_speech_lollms(
            text=text,
            voice=voice,
            model=model,
            response_format=response_format,
            api_url=api_url,
            api_key=api_key,
            audio_sample_path=ref_audio_path,  # Enable voice cloning
            language=language,
        )
    
    elif engine == 'f5':
        from core.resources import manager
        
        if not ref_audio_path:
            raise ValueError("F5-TTS requires a reference audio file")
        
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
        raise ValueError(f"Unknown TTS engine: {engine}. Use 'f5', 'fishspeech', or 'lollms'.")
    

def get_default_tts_engine() -> str:
    """Return safe default TTS engine based on platform and available hardware."""
    # Priority order: xtts (most stable) > f5 (high quality) > fishspeech (fallback)
    
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except:
        has_gpu = False
    
    # XTTS is most stable across platforms
    if has_gpu:
        return 'xtts'
    
    # CPU fallback: use lightweight engines
    if sys.platform == 'win32' or os.name == 'nt':
        return 'piper'  # Fast CPU inference
    
    return 'xtts'  # XTTS works on CPU too, just slower


def generate_speech_xtts(
    text: str,
    ref_audio_path: str,
    device: str = "cuda",
    language: str = "en",
    **kwargs
) -> Tuple[np.ndarray, int]:
    """
    Coqui XTTS v2 - Most stable multilingual voice cloning.
    
    Pros: Very stable, supports 16 languages, good quality
    Cons: Slower than F5, requires more VRAM (~4GB)
    """
    try:
        pm.ensure_packages("TTS>=0.22.0")
        from TTS.api import TTS
    except ImportError:
        raise ImportError(
            "Coqui TTS not installed. Install with:\n"
            "  pip install TTS"
        )
    
    import tempfile
    
    logger.info(f"[XTTS] Generating speech with voice cloning (lang={language})")
    
    # Load model (cached after first load)
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    
    # Clean text for XTTS
    text = text.strip()
    if not text:
        return np.zeros(24000, dtype=np.float32), 24000
    
    # XTTS max length is ~250 chars, split if needed
    max_chars = 250
    chunks = []
    
    if len(text) > max_chars:
        # Split at sentence boundaries
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    else:
        chunks = [text]
    
    # Generate audio for each chunk
    audio_chunks = []
    
    for i, chunk in enumerate(chunks):
        logger.info(f"[XTTS] Processing chunk {i+1}/{len(chunks)}: '{chunk[:50]}...'")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            try:
                tts.tts_to_file(
                    text=chunk,
                    speaker_wav=ref_audio_path,
                    language=language,
                    file_path=tmp.name
                )
                
                # Load generated audio
                audio, sr = sf.read(tmp.name)
                
                # Ensure mono
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                
                # Resample to 24kHz
                if sr != 24000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
                
                audio_chunks.append(audio)
                
            finally:
                try:
                    os.unlink(tmp.name)
                except:
                    pass
    
    # Concatenate chunks with small silence
    if len(audio_chunks) > 1:
        silence = np.zeros(int(0.2 * 24000))  # 200ms silence
        audio = np.concatenate([
            chunk if i == 0 else np.concatenate([silence, chunk])
            for i, chunk in enumerate(audio_chunks)
        ])
    else:
        audio = audio_chunks[0]
    
    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0.95:
        audio = audio / peak * 0.95
    elif peak < 0.1 and peak > 0:
        audio = audio / peak * 0.5
    
    logger.info(f"[XTTS] Generated {len(audio)/24000:.1f}s audio")
    
    return audio.astype(np.float32), 24000


def generate_speech_bark(
    text: str,
    device: str = "cuda",
    voice_preset: str = "v2/en_speaker_6",
    **kwargs
) -> Tuple[np.ndarray, int]:
    """
    Suno Bark - Creative, emotional speech with sound effects.
    
    Pros: Can do emotions, laughter, music; creative
    Cons: No voice cloning, less controllable, slower
    
    Voice presets: v2/en_speaker_0 through v2/en_speaker_9
    """
    try:
        from bark import SAMPLE_RATE, generate_audio, preload_models
    except ImportError:
        raise ImportError(
            "Bark not installed. Install with:\n"
            "  pip install git+https://github.com/suno-ai/bark.git"
        )
    
    logger.info(f"[Bark] Generating speech with preset {voice_preset}")
    
    # Load models (cached)
    preload_models()
    
    # Bark supports special tokens for emotions
    # [laughter], [laughs], [sighs], [music], [gasps], [clears throat]
    # CAPITALIZATION = emphasis
    
    # Clean text
    text = text.strip()
    if not text:
        return np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE
    
    # Bark max length ~14 seconds of speech (~200 chars)
    max_chars = 200
    chunks = []
    
    if len(text) > max_chars:
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    else:
        chunks = [text]
    
    # Generate chunks
    audio_chunks = []
    
    for i, chunk in enumerate(chunks):
        logger.info(f"[Bark] Chunk {i+1}/{len(chunks)}: '{chunk[:50]}...'")
        
        # Add voice preset to text
        chunk_with_preset = f"[{voice_preset}] {chunk}"
        
        audio = generate_audio(chunk_with_preset)
        audio_chunks.append(audio)
    
    # Concatenate
    if len(audio_chunks) > 1:
        silence = np.zeros(int(0.3 * SAMPLE_RATE))
        audio = np.concatenate([
            chunk if i == 0 else np.concatenate([silence, chunk])
            for i, chunk in enumerate(audio_chunks)
        ])
    else:
        audio = audio_chunks[0]
    
    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0.95:
        audio = audio / peak * 0.95
    
    logger.info(f"[Bark] Generated {len(audio)/SAMPLE_RATE:.1f}s audio")
    
    return audio.astype(np.float32), SAMPLE_RATE


def generate_speech_styletts2(
    text: str,
    ref_audio_path: str,
    device: str = "cuda",
    **kwargs
) -> Tuple[np.ndarray, int]:
    """
    StyleTTS2 - Fast, high-quality voice cloning.
    
    Pros: Fast inference, good quality, low VRAM
    Cons: Less robust than XTTS, English-only
    """
    try:
        from styletts2 import tts
    except ImportError:
        raise ImportError(
            "StyleTTS2 not installed. Install with:\n"
            "  pip install git+https://github.com/yl4579/StyleTTS2.git"
        )
    
    logger.info("[StyleTTS2] Generating speech")
    
    # Initialize model (cached)
    model = tts.StyleTTS2()
    
    # Clean text
    text = text.strip()
    if not text:
        return np.zeros(24000, dtype=np.float32), 24000
    
    # Load reference
    ref_audio, sr = librosa.load(ref_audio_path, sr=24000, mono=True)
    
    # Generate
    audio = model.inference(
        text=text,
        ref_s=ref_audio,
        alpha=0.3,  # Style mixing weight
        beta=0.7,   # Content preservation
        diffusion_steps=10,
        embedding_scale=1.0
    )
    
    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0.95:
        audio = audio / peak * 0.95
    
    logger.info(f"[StyleTTS2] Generated {len(audio)/24000:.1f}s audio")
    
    return audio.astype(np.float32), 24000


def generate_speech_piper(
    text: str,
    voice: str = "en_US-lessac-medium",
    **kwargs
) -> Tuple[np.ndarray, int]:
    """
    Piper TTS - Ultra-fast CPU inference (no voice cloning).
    
    Pros: Extremely fast CPU inference, many voices, low resource
    Cons: No voice cloning, lower quality than neural models
    
    Popular voices:
        - en_US-lessac-medium (male, clear)
        - en_US-amy-medium (female, clear)
        - en_GB-alan-medium (British male)
    """
    try:
        import subprocess
        import shutil as sh
    except ImportError:
        raise ImportError("subprocess or shutil not available")
    
    logger.info(f"[Piper] Generating speech with voice {voice}")
    
    # Check if piper binary is available
    piper_bin = sh.which("piper")
    if not piper_bin:
        raise RuntimeError(
            "Piper binary not found. Install from:\n"
            "  https://github.com/rhasspy/piper/releases\n"
            "Or: pip install piper-tts"
        )
    
    # Clean text
    text = text.strip()
    if not text:
        return np.zeros(22050, dtype=np.float32), 22050
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as txt_file:
        txt_file.write(text)
        txt_path = txt_file.name
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
        wav_path = wav_file.name
    
    try:
        # Run piper
        cmd = [
            piper_bin,
            "--model", voice,
            "--output_file", wav_path
        ]
        
        with open(txt_path, 'r') as txt:
            result = subprocess.run(
                cmd,
                stdin=txt,
                capture_output=True,
                timeout=30
            )
        
        if result.returncode != 0:
            raise RuntimeError(f"Piper failed: {result.stderr.decode()}")
        
        # Load audio
        audio, sr = sf.read(wav_path)
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample to 24kHz
        if sr != 24000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
            sr = 24000
        
        # Normalize
        peak = np.max(np.abs(audio))
        if peak > 0.95:
            audio = audio / peak * 0.95
        
        logger.info(f"[Piper] Generated {len(audio)/sr:.1f}s audio")
        
        return audio.astype(np.float32), sr
        
    finally:
        try:
            os.unlink(txt_path)
            os.unlink(wav_path)
        except:
            pass


def get_available_voices_lollms(api_url: Optional[str] = None) -> list:
    """
    Fetch available voices from LoLLMs TTS service.
    
    Returns list of voice names (e.g., ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'])
    """
    import requests
    
    url = api_url or os.getenv("LOLLMS_URL", "http://localhost:9600")
    voices_url = url.rstrip("/") + "/lollms/v1/audio/voices"
    
    key = os.getenv("LOLLMS_API_KEY", "")
    headers = {}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    
    try:
        response = requests.get(voices_url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        # LoLLMs returns OpenAI-compatible format: { "voices": ["alloy", "echo", ...] }
        return data.get('voices', ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'])
    except Exception as e:
        print(f"[LoLLMs TTS] Could not fetch voices: {e}")
        # Return standard OpenAI voices as fallback
        return ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
