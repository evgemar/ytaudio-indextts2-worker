"""
RunPod Serverless worker for IndexTTS-2 with reference audio and duration control.
Features:
  - Multilingual TTS with voice cloning
  - Target duration control (key feature)
  - Reference audio via base64 or URL
  - Supported languages: en, zh, ja, es, fr, de, ru, ko
  - Returns audio_base64 + metadata
"""
import base64
import os
import tempfile
import urllib.request
from pathlib import Path

import runpod
import torch
import torchaudio
import yt_dlp

# Import IndexTTS-2
try:
    from indextts.infer_v2 import IndexTTS2
    print("Successfully imported IndexTTS2 from indextts.infer_v2")
except ImportError as e:
    print(f"Failed to import IndexTTS2: {e}")
    raise ImportError("Could not import IndexTTS2")

MODEL = None
SUPPORTED_LANGUAGES = ['en', 'zh', 'ja', 'es', 'fr', 'de', 'ru', 'ko']
DEFAULT_REF_DURATION = 60  # seconds


def initialize_model():
    """Initialize IndexTTS-2 model"""
    global MODEL
    if MODEL is None:
        print("Initializing IndexTTS2 model...")
        try:
            MODEL = IndexTTS2(
                model_path="IndexTeam/IndexTTS-2",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            print(f"IndexTTS2 initialized on {MODEL.device}")
        except Exception as e:
            print(f"Failed to initialize IndexTTS2: {e}")
            raise
    return MODEL


def load_audio_from_base64(audio_base64: str) -> tuple:
    """Load audio from base64 string"""
    audio_data = base64.b64decode(audio_base64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name

    try:
        waveform, sample_rate = torchaudio.load(tmp_path)
        return waveform, sample_rate
    finally:
        os.unlink(tmp_path)


def load_audio_from_url(audio_url: str) -> tuple:
    """Load audio from URL"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        urllib.request.urlretrieve(audio_url, tmp.name)
        tmp_path = tmp.name

    try:
        waveform, sample_rate = torchaudio.load(tmp_path)
        return waveform, sample_rate
    finally:
        os.unlink(tmp_path)


def synthesize_tts(
    text: str,
    language: str = "en",
    reference_audio=None,
    target_duration_seconds: float = None,
    temperature: float = 1.0,
    speed: float = 1.0
) -> tuple:
    """
    Synthesize speech using IndexTTS-2

    Returns:
        tuple: (audio_tensor, sample_rate)
    """
    model = initialize_model()

    # Validate language
    if language not in SUPPORTED_LANGUAGES:
        print(f"Warning: Language {language} not in supported list, using 'en'")
        language = "en"

    try:
        # Synthesize with IndexTTS-2
        result = model.synthesize(
            text=text,
            language=language,
            reference_audio=reference_audio,
            target_duration=target_duration_seconds,
            temperature=temperature,
            speed=speed
        )

        # Extract audio tensor and sample rate
        if isinstance(result, dict):
            audio = result.get('audio')
            sr = result.get('sample_rate', 24000)
        else:
            audio = result
            sr = 24000  # Default sample rate

        return audio, sr

    except Exception as e:
        print(f"Synthesis failed: {e}")
        raise


def handler(job):
    """Main RunPod handler"""
    job_input = job["input"]

    # Extract required parameters
    text = job_input.get("text")
    if not text:
        return {"error": "Missing required parameter: text"}

    language = job_input.get("language", "en")
    target_duration_seconds = job_input.get("target_duration_seconds")
    temperature = float(job_input.get("temperature", 1.0))
    speed = float(job_input.get("speed", 1.0))

    # Load reference audio
    reference_audio = None
    if "reference_audio_base64" in job_input:
        try:
            ref_audio, ref_sr = load_audio_from_base64(job_input["reference_audio_base64"])
            reference_audio = (ref_audio, ref_sr)
        except Exception as e:
            return {"error": f"Failed to load reference audio from base64: {e}"}
    elif "audio_url" in job_input:
        try:
            ref_audio, ref_sr = load_audio_from_url(job_input["audio_url"])
            reference_audio = (ref_audio, ref_sr)
        except Exception as e:
            return {"error": f"Failed to load reference audio from URL: {e}"}
    else:
        return {"error": "Missing reference audio: provide 'reference_audio_base64' or 'audio_url'"}

    try:
        # Synthesize speech
        audio_tensor, sample_rate = synthesize_tts(
            text=text,
            language=language,
            reference_audio=reference_audio,
            target_duration_seconds=target_duration_seconds,
            temperature=temperature,
            speed=speed
        )

        # Convert to audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            torchaudio.save(tmp.name, audio_tensor, sample_rate)
            tmp_path = tmp.name

        try:
            # Read audio file and encode to base64
            with open(tmp_path, "rb") as f:
                audio_data = f.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        finally:
            os.unlink(tmp_path)

        # Return result
        return {
            "audio_base64": audio_base64,
            "sample_rate": sample_rate,
            "metadata": {
                "text": text,
                "language": language,
                "target_duration_seconds": target_duration_seconds,
                "actual_duration_seconds": len(audio_tensor[0]) / sample_rate if len(audio_tensor.shape) > 1 else len(audio_tensor) / sample_rate,
                "temperature": temperature,
                "speed": speed,
                "supported_languages": SUPPORTED_LANGUAGES
            }
        }

    except Exception as e:
        return {"error": f"Synthesis failed: {str(e)}"}


if __name__ == "__main__":
    print(f"IndexTTS2 Worker - Supported languages: {SUPPORTED_LANGUAGES}")
    print(f"MULTILINGUAL=true")
    runpod.serverless.start({"handler": handler})