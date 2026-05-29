# IndexTTS-2 RunPod Serverless Worker

RunPod Serverless worker for **IndexTTS-2** text-to-speech with voice cloning and duration control.

## Features

- **Multilingual TTS**: Supports en, zh, ja, es, fr, de, ru, ko
- **Voice Cloning**: Reference audio via base64 or URL
- **Duration Control**: Target duration for precise timing (key feature)
- **High Quality**: IndexTTS-2 model for natural speech synthesis

## Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | ✅ | Text to synthesize |
| `language` | string | ❌ | Language code (default: "en") |
| `reference_audio_base64` | string | ⚠️ | Base64-encoded speaker / timbre reference audio (spk_audio_prompt) |
| `audio_url` | string | ⚠️ | URL to speaker / timbre reference audio |
| `emotion_reference_base64` | string | ❌ | Base64-encoded emotion / manner reference audio (emo_audio_prompt) |
| `emotion_audio_url` | string | ❌ | URL to emotion / manner reference audio |
| `emo_alpha` | float | ❌ | Emotion blending strength (default: 1.0) |
| `target_duration_seconds` | float | ❌ | Accepted for metadata only — IndexTTS-2 has no native duration arg; callers must apply atempo/duration matching themselves |
| `temperature` | float | ❌ | Synthesis temperature (default: 1.0) |
| `speed` | float | ❌ | Speech speed (default: 1.0) |

⚠️ Either `reference_audio_base64` or `audio_url` is required for the speaker
reference. Emotion reference is optional — when omitted the worker behaves
exactly as before (timbre-only clone).

## Output

```json
{
  "audio_base64": "UklGRi...",
  "sample_rate": 24000,
  "metadata": {
    "text": "Hello world",
    "language": "en",
    "target_duration_seconds": 2.5,
    "actual_duration_seconds": 2.48,
    "temperature": 1.0,
    "speed": 1.0,
    "supported_languages": ["en", "zh", "ja", "es", "fr", "de", "ru", "ko"]
  }
}
```

## Deployment

1. Build the Docker image:
```bash
docker build -t indextts2-worker .
```

2. Deploy to RunPod Serverless:
   - Upload image to Docker registry
   - Create new serverless endpoint in RunPod console
   - Use this image in endpoint configuration

3. Get your endpoint ID from RunPod console

4. Configure in ytaudio `.env`:
```env
INDEXTTS2_BASE_URL=https://api.runpod.ai/v2/YOUR_ENDPOINT_ID
INDEXTTS2_API_KEY=your_runpod_api_key
```

## Integration with ytaudio

The IndexTTS-2 provider automatically:
- Routes English target language jobs to IndexTTS-2
- Passes the per-segment emotion reference for manner transfer
- Provides fallback to Chatterbox if IndexTTS-2 fails

### Verified `IndexTTS2.infer()` signature (index-tts main)

```python
infer(self, spk_audio_prompt, text, output_path,
      emo_audio_prompt=None, emo_alpha=1.0,
      emo_vector=None,
      use_emo_text=False, emo_text=None,
      use_random=False, interval_silence=200,
      verbose=False, max_text_tokens_per_segment=120,
      stream_return=False, more_segment_before=0,
      **generation_kwargs)
```

There is no duration-control parameter in this release; duration matching
remains a caller responsibility (ffmpeg atempo).

## Supported Languages

- English (en)
- Chinese (zh) 
- Japanese (ja)
- Spanish (es)
- French (fr)
- German (de)
- Russian (ru)
- Korean (ko)

## Performance

- Cold start: ~10-15 seconds (model preloaded)
- Synthesis: ~2-5 seconds per segment
- Duration control: Precise timing without pitch distortion

