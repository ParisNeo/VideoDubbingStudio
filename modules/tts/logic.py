# modules/tts/logic.py
import os, io, base64, tempfile
import numpy as np
import soundfile as sf
import librosa
import requests

from core.resources import manager
from f5_tts.infer.utils_infer import infer_process

FISH_SPEECH_API_URL = os.getenv("FISH_SPEECH_API_URL", "http://127.0.0.1:8080/v1/tts")

def _wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    bio = io.BytesIO()
    sf.write(bio, audio.astype(np.float32), sr, format="WAV")
    return bio.getvalue()

def _normalize_audio(audio: np.ndarray, target_level: float = -3.0) -> np.ndarray:
    """
    Soft‑clip and scale audio to a target RMS level (in dBFS).
    This prevents clipping while preserving dynamics.
    """
    if audio.size == 0:
        return audio
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-6:
        return audio
    # Convert target dBFS to linear gain
    target_gain = 10 ** (target_level / 20)
    scalar = target_gain / (rms + 1e-9)
    # Soft clipping to avoid harsh distortion
    scaled = audio * scalar
    return np.tanh(scaled) * 0.99

def _fishspeech_tts(text: str, ref_audio: np.ndarray, ref_sr: int, ref_text: str) -> tuple[np.ndarray, int]:
    payload = {
        "text": text,
        "reference_audio": base64.b64encode(_wav_bytes(ref_audio, ref_sr)).decode("utf-8"),
        "reference_text": ref_text or "",
    }
    r = requests.post(FISH_SPEECH_API_URL, json=payload, timeout=300)

    ct = (r.headers.get("content-type") or "").lower()
    if "application/json" in ct:
        j = r.json()
        b64 = j.get("audio") or j.get("wav") or j.get("data") or j.get("audio_base64")
        if not b64:
            raise RuntimeError(f"FishSpeech returned JSON without audio field: keys={list(j.keys())}")
        audio_bytes = base64.b64decode(b64)
    else:
        audio_bytes = r.content

    # Write to a temp file and load with librosa for consistency
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tf.write(audio_bytes)
        tmp_path = tf.name
    try:
        audio, sr = librosa.load(tmp_path, sr=None, mono=True)
        # Normalise the returned audio to avoid saturation
        audio = _normalize_audio(audio)
        return audio, sr
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def generate_speech(text: str, ref_audio_path: str, engine: str = "f5"):
    engine = (engine or "f5").lower()

    # Load reference audio (keep a reasonable length and mono)
    ref_audio, ref_sr = librosa.load(ref_audio_path, sr=24000, mono=True)
    ref_audio = ref_audio[: 10 * 24000]  # limit to first 10 s for stability

    # Optional ASR for Fish Speech reference text
    if engine == "fishspeech":
        stt = manager.get_whisper()
        asr_res = stt(ref_audio_path, generate_kwargs={"task": "transcribe"})
        ref_text = (asr_res.get("text") or "").strip()
        audio, sr = _fishspeech_tts(text=text, ref_audio=ref_audio, ref_sr=ref_sr, ref_text=ref_text)
        return audio, sr

    if engine == "f5":
        f5_model, vocoder = manager.get_f5_tts()
        # Write reference audio to a temporary wav for the infer_process API
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            sf.write(tf.name, ref_audio, 24000)
            ref_tmp = tf.name
        try:
            gen_audio, sr, _ = infer_process(
                ref_audio=ref_tmp,
                ref_text="",  # no explicit reference text for F5‑TTS
                gen_text=text,
                model_obj=f5_model,
                vocoder=vocoder,
                mel_spec_type="vocos",
                device=manager.device,
            )
            # Convert to numpy array if needed
            if hasattr(gen_audio, "cpu"):
                gen_audio = gen_audio.cpu().numpy()
            gen_audio = np.asarray(gen_audio).flatten()
            # Normalise to avoid clipping / saturation
            gen_audio = _normalize_audio(gen_audio)
            return gen_audio, sr
        finally:
            try:
                os.remove(ref_tmp)
            except Exception:
                pass

    raise ValueError(f"Unknown engine: {engine}")
