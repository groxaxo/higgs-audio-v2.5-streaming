# Audio Codec Configuration

This document explains how to configure audio format conversion in the Higgs Audio TTS API.

## Overview

Audio encoding/format conversion (MP3, Opus, AAC, FLAC, WAV) is performed on the CPU to ensure consistent behavior across CPU and GPU deployments. The system supports two encoding backends:

1. **TorchCodec** - PyTorch's native audio codec support (optional)
2. **FFmpeg** - Traditional audio conversion tool (fallback)

## Configuration

The audio codec backend is controlled via the `AUDIO_CODEC` environment variable:

### Environment Variable Values

| Value | Behavior |
|-------|----------|
| `auto` (default) | Use TorchCodec if available, otherwise fall back to FFmpeg |
| `torchcodec` | Always attempt to use TorchCodec (will fail if not installed) |
| `ffmpeg` | Always use FFmpeg for non-WAV/FLAC formats |

**Note:** The environment variable is case-insensitive.

## Deployment Recommendations

### GPU Deployments

```yaml
environment:
  - AUDIO_CODEC=auto
```

This allows the system to use TorchCodec if installed, providing the best performance.

### CPU-Only Deployments

```yaml
environment:
  - AUDIO_CODEC=ffmpeg
```

For CPU-only deployments where TorchCodec may not be installed, explicitly using FFmpeg ensures reliable operation.

### Development/Testing

```yaml
environment:
  - AUDIO_CODEC=torchcodec  # Test TorchCodec path
  # or
  - AUDIO_CODEC=ffmpeg      # Test FFmpeg path
```

## How It Works

1. **Audio generation** happens on CPU or GPU (your choice via `DEVICE` env var)
2. **Format conversion** always moves tensors to CPU for encoding
3. The codec backend is selected based on `AUDIO_CODEC`:
   - WAV/FLAC: Always use torchaudio directly (no codec needed)
   - MP3/Opus/AAC: Use TorchCodec or FFmpeg based on configuration

## Docker Configuration

The `docker-compose.yml` includes the `AUDIO_CODEC` environment variable by default:

```yaml
services:
  higgs-audio-tts:
    environment:
      - AUDIO_CODEC=auto  # auto | torchcodec | ffmpeg
```

For CPU-only variant:

```yaml
  higgs-audio-tts-cpu:
    environment:
      - AUDIO_CODEC=ffmpeg  # Use FFmpeg for CPU-only deployments
```

## Benefits

- **Consistent behavior**: Same conversion logic for CPU and GPU deployments
- **Flexibility**: Choose the right backend for your environment
- **Reliability**: Automatic fallback ensures the system works even without TorchCodec
- **No "works on CPU but breaks on GPU" issues**: Conversion is always on CPU

## Format Support

| Format | Direct Support | Requires Codec |
|--------|---------------|----------------|
| WAV | ✓ (torchaudio) | No |
| FLAC | ✓ (torchaudio) | No |
| MP3 | Via codec | Yes (TorchCodec/FFmpeg) |
| Opus | Via codec | Yes (TorchCodec/FFmpeg) |
| AAC | Via codec | Yes (TorchCodec/FFmpeg) |

## Troubleshooting

### MP3/Opus/AAC conversion fails

**Symptom**: Error when requesting non-WAV/FLAC formats

**Solutions**:
1. Ensure FFmpeg is installed in your container/system
2. Set `AUDIO_CODEC=ffmpeg` to force FFmpeg usage
3. Check that FFmpeg is in your PATH

### TorchCodec errors

**Symptom**: Errors mentioning TorchCodec

**Solutions**:
1. Set `AUDIO_CODEC=ffmpeg` to bypass TorchCodec
2. Install TorchCodec if you want to use it
3. Use `AUDIO_CODEC=auto` to automatically fall back to FFmpeg

## Example Usage

```bash
# Use auto-detection (recommended)
docker-compose up

# Force FFmpeg
AUDIO_CODEC=ffmpeg docker-compose up

# Force TorchCodec (if installed)
AUDIO_CODEC=torchcodec docker-compose up
```

## Testing

The codec switching logic is tested in `api/tests/test_audio_codec.py`. Run tests with:

```bash
python -m pytest api/tests/test_audio_codec.py -v
```
