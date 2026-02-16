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


def generate_speech_lollms(
    text: str,
    voice: Optional[str] = None,
    model: Optional[str] = None,
    response_format: str = "wav",  # Changed default to wav for better compatibility
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    audio_sample_path: Optional[str] = None,
    max_retries: int = 2,
) -> Tuple[np.ndarray, int]:
    """
    LoLLMs TTS API-based speech generation.
    
    Uses the OpenAI-compatible /v1/audio/speech endpoint (more reliable than /lollms/v1).
    Voice cloning is handled via the reference_audio parameter if the endpoint supports it.
    
    Args:
        text: Text to synthesize
        voice: Voice name (alloy, echo, fable, onyx, nova, shimmer)
        model: Model name (tts-1, tts-1-hd)
        response_format: Audio format (wav recommended for direct loading)
        api_url: LoLLMs API URL (default: http://localhost:9642)
        api_key: LoLLMs API key
        audio_sample_path: Path to reference audio for voice cloning (optional)
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
    
    # Get LoLLMs configuration
    base_url = api_url or os.getenv("LOLLMS_URL", "http://localhost:9642")
    key = api_key or os.getenv("LOLLMS_API_KEY", "")
    
    # Try OpenAI-compatible endpoint first (more reliable)
    # Fall back to /lollms/v1 if needed
    urls_to_try = [
        base_url.rstrip("/") + "/v1/audio/speech",  # OpenAI compatible
        base_url.rstrip("/") + "/lollms/v1/audio/speech",  # LoLLMs native
    ]
    
    voice = voice or "alloy"
    model = model or "tts-1"
    
    headers = {
        "Content-Type": "application/json",
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"
    
    # Build OpenAI-compatible payload
    payload = {
        "input": text,
        "voice": voice,
        "model": model,
        "response_format": response_format
    }
    
    # Try voice cloning if reference audio provided
    # Note: This is non-standard extension, may not work with all LoLLMs versions
    if audio_sample_path and Path(audio_sample_path).exists():
        try:
            with open(audio_sample_path, "rb") as f:
                ref_bytes = f.read()
                # Some versions accept reference_audio as base64
                payload["reference_audio"] = base64.b64encode(ref_bytes).decode('utf-8')
        except Exception as e:
            print(f"[LoLLMs TTS] Warning: Could not load reference audio: {e}")
    
    last_error = None
    
    for attempt in range(max_retries):
        for url in urls_to_try:
            try:
                print(f"[LoLLMs TTS] Attempt {attempt + 1}/{max_retries} to {url}")
                
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                
                # Log error details for debugging
                if response.status_code != 200:
                    error_detail = ""
                    try:
                        error_detail = response.json()
                    except:
                        error_detail = response.text[:500]
                    print(f"[LoLLMs TTS] Error {response.status_code}: {error_detail}")
                    continue  # Try next URL
                
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
                        # WAV/FLAC - load directly
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
                continue  # Try next URL
            except requests.exceptions.Timeout as e:
                last_error = f"Timeout connecting to {url}: {e}"
                print(f"[LoLLMs TTS] {last_error}")
                continue
            except Exception as e:
                last_error = f"Error with {url}: {e}"
                print(f"[LoLLMs TTS] {last_error}")
                continue
        
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
        f"2) TTS service is enabled in LoLLMs settings, "
        f"3) Try using 'f5' or 'fishspeech' engine instead."
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
    
    Args:
        text: Text to synthesize
        ref_audio_path: Path to reference audio file (required for F5/FishSpeech/LoLLMs voice cloning)
        engine: 'f5', 'fishspeech', 'lollms', or None for platform-aware default
        device: Preferred compute device (ignored for FishSpeech and LoLLMs)
    
    Returns:
        (audio_array, sample_rate)
    """
    # Auto-select engine if not specified
    if engine is None:
        engine = 'fishspeech' if os.name == 'nt' else 'f5'
    
    engine = engine.lower()
    
    if engine == 'fishspeech':
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
        return generate_speech_lollms(
            text=text,
            voice=voice,
            model=model,
            response_format=response_format,
            api_url=api_url,
            api_key=api_key,
            audio_sample_path=ref_audio_path,  # Enable voice cloning
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
    """Return safe default TTS engine. Windows uses FishSpeech to avoid SVML."""
    if sys.platform == 'win32' or os.name == 'nt':
        return 'fishspeech'
    
    # Linux/Mac: use F5-TTS if GPU available
    try:
        import torch
        if torch.cuda.is_available():
            return 'f5'
    except:
        pass
    
    return 'fishspeech'


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
