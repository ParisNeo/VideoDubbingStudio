"""
F5-TTS Diagnostic Tool
Tests F5-TTS with a simple reference audio to diagnose gibberish issues.

Usage:
    python test_f5_tts.py path/to/reference.wav "Text to synthesize"
"""

import sys
import os
import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.tts.logic import generate_speech_f5_robust
from core.resources import manager

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_f5_tts.py <reference_audio.wav> <text_to_synthesize>")
        print("\nExample:")
        print('  python test_f5_tts.py sample.wav "Hello, this is a test."')
        sys.exit(1)
    
    ref_audio_path = sys.argv[1]
    text = sys.argv[2]
    
    if not Path(ref_audio_path).exists():
        print(f"ERROR: Reference audio file not found: {ref_audio_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("F5-TTS Diagnostic Test")
    print("=" * 60)
    print(f"Reference Audio: {ref_audio_path}")
    print(f"Text to Synthesize: {text}")
    print("=" * 60)
    
    # Load models
    print("\n[1/4] Loading F5-TTS models...")
    try:
        f5_model, vocoder = manager.get_f5_tts()
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        sys.exit(1)
    
    # Analyze reference audio
    print("\n[2/4] Analyzing reference audio...")
    try:
        audio, sr = librosa.load(ref_audio_path, sr=24000, mono=True)
        duration = len(audio) / sr
        peak = np.max(np.abs(audio))
        std = np.std(audio)
        
        print(f"  Duration: {duration:.2f}s")
        print(f"  Sample Rate: {sr} Hz")
        print(f"  Peak Amplitude: {peak:.4f}")
        print(f"  Std Deviation: {std:.4f}")
        
        if duration < 3.0:
            print("  ⚠️  WARNING: Reference audio is short (<3s), quality may suffer")
        if duration > 15.0:
            print("  ⚠️  WARNING: Reference audio is long (>15s), will be trimmed")
        if peak < 0.1:
            print("  ⚠️  WARNING: Reference audio is very quiet")
        if peak > 0.99:
            print("  ⚠️  WARNING: Reference audio may be clipped")
        if std < 0.01:
            print("  ⚠️  WARNING: Reference audio has low variance (possible silence)")
    except Exception as e:
        print(f"❌ Failed to analyze reference: {e}")
        sys.exit(1)
    
    # Generate speech
    print("\n[3/4] Generating speech with F5-TTS...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Using device: {device}")
        
        audio_out, sr_out = generate_speech_f5_robust(
            text=text,
            ref_audio_path=ref_audio_path,
            model_obj=f5_model,
            vocoder=vocoder,
            device=device,
            max_retries=3
        )
        
        print("✅ Speech generated successfully")
        
        # Analyze output
        out_duration = len(audio_out) / sr_out
        out_peak = np.max(np.abs(audio_out))
        out_std = np.std(audio_out)
        
        print(f"  Output Duration: {out_duration:.2f}s")
        print(f"  Output Peak: {out_peak:.4f}")
        print(f"  Output Std: {out_std:.4f}")
        
        # Quality checks
        if out_std < 0.001:
            print("  ❌ CRITICAL: Output is near-silent (likely gibberish)")
        elif out_peak > 5.0:
            print("  ❌ CRITICAL: Output is clipped (likely gibberish)")
        elif not np.isfinite(audio_out).all():
            print("  ❌ CRITICAL: Output contains NaN/Inf values")
        else:
            print("  ✅ Output passes quality checks")
        
    except Exception as e:
        print(f"❌ Failed to generate speech: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save output
    output_path = "test_f5_output.wav"
    print(f"\n[4/4] Saving output to {output_path}...")
    try:
        sf.write(output_path, audio_out, sr_out)
        print(f"✅ Saved: {output_path}")
        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)
        print(f"Listen to the output file to verify quality:")
        print(f"  {output_path}")
    except Exception as e:
        print(f"❌ Failed to save: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()