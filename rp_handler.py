"""
RunPod Serverless worker for IndexTTS-2 with reference audio and emotion transfer.

Verified IndexTTS2.infer() signature (indextts.infer_v2.IndexTTS2):
    infer(self, spk_audio_prompt, text, output_path,
          emo_audio_prompt=None, emo_alpha=1.0,
          emo_vector=None,
          use_emo_text=False, emo_text=None,
          use_random=False, interval_silence=200,
          verbose=False, max_text_tokens_per_segment=120,
          stream_return=False, more_segment_before=0,
          **generation_kwargs)

Notes:
  - There is NO native duration-control argument in this IndexTTS-2 release.
    `interval_silence` only controls between-segment silence in ms.
    `target_duration_seconds` from job_input is accepted for forward
    compatibility and echoed in metadata but is NOT passed to .infer();
    callers must continue to handle duration matching (e.g. ffmpeg atempo).
  - Backward compatible: if no emotion reference is provided, behaves
    exactly as before (timbre-only via spk_audio_prompt).
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
                cfg_path="/app/checkpoints/config.yaml",
                model_dir="/app/checkpoints",
                use_fp16=torch.cuda.is_available(),
            )
            print("IndexTTS2 initialized")
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


def _write_ref_to_path(reference_audio, path: str) -> str:
    """Coerce a reference audio input (tuple|str path) to a file path on disk.

    Returns the path that should be passed to model.infer().
    """
    if isinstance(reference_audio, tuple):
        ref_audio, ref_sr = reference_audio
        if ref_audio.dim() == 1:
            ref_audio = ref_audio.unsqueeze(0)
        torchaudio.save(path, ref_audio, ref_sr, format="wav")
        return path
    if isinstance(reference_audio, str):
        return reference_audio  # already a path on disk
    raise RuntimeError("reference audio missing or wrong type")


def synthesize_tts(
    text: str,
    language: str = "en",
    reference_audio=None,
    emotion_reference_audio=None,
    emo_alpha: float = 1.0,
    target_duration_seconds: float = None,
    temperature: float = 1.0,
    speed: float = 1.0,
) -> tuple:
    """
    Synthesize speech using IndexTTS-2.

    - reference_audio: speaker / timbre clone reference (spk_audio_prompt)
    - emotion_reference_audio: optional emotion / manner reference
      (emo_audio_prompt). When None, infer() will use the speaker prompt's
      emotion as before.
    - emo_alpha: emotion blending strength (0..1+), default 1.0
    - target_duration_seconds: not natively supported by this IndexTTS-2
      release; caller must apply post-processing duration matching.

    Returns:
        tuple: (audio_tensor, sample_rate)
    """
    model = initialize_model()

    if language not in SUPPORTED_LANGUAGES:
        print(f"Warning: Language {language} not in supported list, using 'en'")
        language = "en"

    try:
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "gen.wav")
            spk_path = _write_ref_to_path(reference_audio, os.path.join(td, "ref_spk.wav"))

            emo_path = None
            if emotion_reference_audio is not None:
                emo_path = _write_ref_to_path(
                    emotion_reference_audio, os.path.join(td, "ref_emo.wav")
                )

            infer_kwargs = dict(
                spk_audio_prompt=spk_path,
                text=text,
                output_path=out_path,
                verbose=False,
            )
            if emo_path is not None:
                infer_kwargs["emo_audio_prompt"] = emo_path
                infer_kwargs["emo_alpha"] = float(emo_alpha)

            model.infer(**infer_kwargs)
            audio_t, sr = torchaudio.load(out_path)
            return audio_t, sr

    except Exception as e:
        print(f"Synthesis failed: {e}")
        raise


def _load_reference(job_input, base64_key: str, url_key: str):
    """Load a reference audio bundle (tensor, sr) from either base64 or URL.

    Returns (audio_tuple, error_string). Either is None if the other is set.
    Returns (None, None) if neither key is present.
    """
    if base64_key in job_input:
        try:
            wav, sr = load_audio_from_base64(job_input[base64_key])
            return (wav, sr), None
        except Exception as e:
            return None, f"Failed to load {base64_key}: {e}"
    if url_key in job_input:
        try:
            wav, sr = load_audio_from_url(job_input[url_key])
            return (wav, sr), None
        except Exception as e:
            return None, f"Failed to load {url_key}: {e}"
    return None, None


def handler(job):
    """Main RunPod handler"""
    job_input = job["input"]

    text = job_input.get("text")
    if not text:
        return {"error": "Missing required parameter: text"}

    language = job_input.get("language", "en")
    target_duration_seconds = job_input.get("target_duration_seconds")
    temperature = float(job_input.get("temperature", 1.0))
    speed = float(job_input.get("speed", 1.0))
    emo_alpha = float(job_input.get("emo_alpha", 1.0))

    # Speaker / timbre reference (required) — backward compatible keys
    reference_audio, err = _load_reference(
        job_input, "reference_audio_base64", "audio_url"
    )
    if err:
        return {"error": err}
    if reference_audio is None:
        return {
            "error": "Missing reference audio: provide 'reference_audio_base64' or 'audio_url'"
        }

    # Emotion / manner reference (optional)
    emotion_reference_audio, err = _load_reference(
        job_input, "emotion_reference_base64", "emotion_audio_url"
    )
    if err:
        return {"error": err}

    try:
        audio_tensor, sample_rate = synthesize_tts(
            text=text,
            language=language,
            reference_audio=reference_audio,
            emotion_reference_audio=emotion_reference_audio,
            emo_alpha=emo_alpha,
            target_duration_seconds=target_duration_seconds,
            temperature=temperature,
            speed=speed,
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            torchaudio.save(tmp.name, audio_tensor, sample_rate)
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as f:
                audio_data = f.read()
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
        finally:
            os.unlink(tmp_path)

        actual_duration = (
            len(audio_tensor[0]) / sample_rate
            if len(audio_tensor.shape) > 1
            else len(audio_tensor) / sample_rate
        )

        return {
            "audio_base64": audio_base64,
            "sample_rate": sample_rate,
            "metadata": {
                "text": text,
                "language": language,
                "target_duration_seconds": target_duration_seconds,
                "actual_duration_seconds": actual_duration,
                "temperature": temperature,
                "speed": speed,
                "emo_alpha": emo_alpha,
                "used_emotion_reference": emotion_reference_audio is not None,
                "duration_control_supported": False,
                "supported_languages": SUPPORTED_LANGUAGES,
            },
        }

    except Exception as e:
        return {"error": f"Synthesis failed: {str(e)}"}


if __name__ == "__main__":
    print(f"IndexTTS2 Worker - Supported languages: {SUPPORTED_LANGUAGES}")
    print("MULTILINGUAL=true")
    runpod.serverless.start({"handler": handler})
