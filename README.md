<h1 align="center">Higgs Audio: Redefining Expressiveness in Audio Generation</h1>

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://boson.ai/blog/higgs-audio-v2"><img src='https://img.shields.io/badge/🚀-V2 Blogpost-228B22' style="margin-right: 5px;"></a>
  <a href="https://www.boson.ai/blog/higgs-audio-v2.5"><img src='https://img.shields.io/badge/🚀-V2.5 Blogpost-228B22' style="margin-right: 5px;"></a>
  <a href="https://boson.ai/demo/tts"><img src="https://img.shields.io/badge/🕹️-Boson%20AI%20Playground-9C276A" style="margin-right: 5px;"></a>
  <a href="https://huggingface.co/spaces/smola/higgs_audio_v2"><img src="https://img.shields.io/badge/🎮-HF%20Space%20Playground-8A2BE2" style="margin-right: 5px;"></a>
  <a href="https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base"><img src="https://img.shields.io/badge/🤗-Checkpoints (3.6B LLM + 2.2B audio adapter)-ED5A22.svg" style="margin-right: 5px;"></a>
</div>

## NEWS！

### 🎙️ NEW: OpenAI-Compatible TTS Server

We've added an **OpenAI-compatible Text-to-Speech API server** using FastAPI! This allows you to use Higgs Audio with any application that supports OpenAI's TTS API.

**Quick Start:**
```bash
# Using Docker Compose
docker-compose up -d

# Or using the startup script
./start_server.sh
```

Then visit http://localhost:8000 for the web interface or http://localhost:8000/docs for API documentation.

See the [OpenAI-Compatible API](#openai-compatible-api) section for complete usage details.

---

We are proud to launch **Higgs-Audio V2.5**, the latest iteration of Boson AI’s Audio model, designed to bring high-fidelity generation into production environments. Building on Higgs-Audio V2, this release combines improved efficiency with the stability required for real-world deployment.

With V2.5, we condensed the model architecture to 1B parameters while surpassing speed and accuracy of the prior 3B model. The result is achieved through a new alignment strategy using Group Relative Policy Optimization (GRPO) on our curated Voice Bank dataset, combined with improved voice cloning and finer-grained style control.

For detailed model performance, key improvements, and usage, please check our [blog](https://www.boson.ai/blog/higgs-audio-v2.5).


## Higgs Audio V2


We are open-sourcing Higgs Audio v2, a powerful audio foundation model pretrained on over 10 million hours of audio data and a diverse set of text data. Despite having no post-training or fine-tuning, Higgs Audio v2 excels in expressive audio generation, thanks to its deep language and acoustic understanding.

On [EmergentTTS-Eval](https://github.com/boson-ai/emergenttts-eval-public), it achieves win rates of **75.7%** and **55.7%** over "gpt-4o-mini-tts" on the "Emotions" and "Questions" categories, respectively. It also obtains state-of-the-art performance on traditional TTS benchmarks like Seed-TTS Eval and Emotional Speech Dataset (ESD). Moreover, the model demonstrates capabilities rarely seen in previous systems, including generating natural multi-speaker dialogues in multiple languages, automatic prosody adaptation during narration, melodic humming with the cloned voice, and simultaneous generation of speech and background music.

<p align="center">
    <img src="figures/emergent-tts-emotions-win-rate.png" width=900>
</p>

Here's the demo video that shows some of its emergent capabilities (remember to unmute):

<video src="https://github.com/user-attachments/assets/0fd73fad-097f-48a9-9f3f-bc2a63b3818d" type="video/mp4" width="80%" controls>
</video>

Here's another demo video that show-cases the model's multilingual capability and how it enabled live translation (remember to unmute):

<video src="https://github.com/user-attachments/assets/2b9b01ff-67fc-4bd9-9714-7c7df09e38d6" type="video/mp4" width="80%" controls>
</video>

## Installation

We recommend to use NVIDIA Deep Learning Container to manage the CUDA environment. Following are two docker images that we have verified:
- nvcr.io/nvidia/pytorch:25.02-py3
- nvcr.io/nvidia/pytorch:25.01-py3

Here's an example command for launching a docker container environment. Please also check the [official NVIDIA documentations](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

```bash
docker run --gpus all --ipc=host --net=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm nvcr.io/nvidia/pytorch:25.02-py3 bash
```

### Option 1: Direct installation


```bash
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio

pip install -r requirements.txt
pip install -e .
```

### Option 2: Using venv

```bash
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio

python3 -m venv higgs_audio_env
source higgs_audio_env/bin/activate
pip install -r requirements.txt
pip install -e .
```


### Option 3: Using conda
```bash
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio

conda create -y --prefix ./conda_env --override-channels --strict-channel-priority --channel "conda-forge" "python==3.10.*"
conda activate ./conda_env
pip install -r requirements.txt
pip install -e .

# Uninstalling environment:
conda deactivate
conda remove -y --prefix ./conda_env --all
```

### Option 4: Using uv
```bash
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio

uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

### Option 5: Using vllm

For advanced usage with higher throughput, we also built OpenAI compatible API server backed by vLLM engine for you to use.
Please refer to [examples/vllm](./examples/vllm) for more details.


## Usage

> [!TIP]
> For optimal performance, run the generation examples on a machine equipped with GPU with at least 24GB memory!

### Get Started

Here's a basic python snippet to help you get started.

```python
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent

import torch
import torchaudio
import time
import click

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

system_prompt = (
    "Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"
)

messages = [
    Message(
        role="system",
        content=system_prompt,
    ),
    Message(
        role="user",
        content="The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years.",
    ),
]
device = "cuda" if torch.cuda.is_available() else "cpu"

serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)

output: HiggsAudioResponse = serve_engine.generate(
    chat_ml_sample=ChatMLSample(messages=messages),
    max_new_tokens=1024,
    temperature=0.3,
    top_p=0.95,
    top_k=50,
    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
)
torchaudio.save(f"output.wav", torch.from_numpy(output.audio)[None, :], output.sampling_rate)
```

We also provide a list of examples under [examples](./examples). In the following we highlight a few examples to help you use Higgs Audio v2.

## OpenAI-Compatible API

We provide a FastAPI-based server that is **fully compatible with OpenAI's TTS API**. This allows you to use Higgs Audio as a drop-in replacement for OpenAI's TTS service.

### Quick Start

#### Using Docker (Recommended)

```bash
# Build and run with Docker Compose (GPU)
docker-compose up -d

# Or build and run manually
docker build -t higgs-audio-tts .

# Run with GPU support
docker run --gpus all -p 8000:8000 higgs-audio-tts

# Run with CPU only
docker run -p 8000:8000 -e DEVICE=cpu higgs-audio-tts
```

#### Running Directly

```bash
# Quick start with provided script
./start_server.sh

# Or install and run manually
pip install -r requirements.txt
pip install -r api/requirements.txt
pip install -e .

# Start the server
python -m uvicorn api.src.main:app --host 0.0.0.0 --port 8000
```

### API Usage Examples

Once the server is running, you can access:
- **Landing page**: http://localhost:8000/
- **API docs**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health

#### Basic Text-to-Speech

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello, this is a test of the Higgs Audio TTS system.",
    "voice": "alloy",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

#### With Speed Control

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "The quick brown fox jumps over the lazy dog.",
    "voice": "nova",
    "speed": 1.25,
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

#### Using with OpenAI Python SDK

```python
from openai import OpenAI

# Point to your local server
client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Hello world! This is generated by Higgs Audio."
)

response.stream_to_file("output.mp3")
```

#### List Available Voices

```bash
curl http://localhost:8000/v1/voices
```

### Supported Voices

**OpenAI-compatible voices:** `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

**Native Higgs Audio voices:** `af_heart`, `af_nicole`, `af_sarah`, `am_adam`, `am_michael`, `bf_emma`, `bf_isabella`, `bm_george`, `bm_lewis`

### Supported Formats

`mp3`, `opus`, `aac`, `flac`, `wav`

### Configuration

Set these environment variables to configure the server:

- `MODEL_PATH`: Path to the Higgs Audio model (default: `bosonai/higgs-audio-v2-generation-3B-base`)
- `AUDIO_TOKENIZER_PATH`: Path to the audio tokenizer (default: `bosonai/higgs-audio-v2-tokenizer`)
- `DEVICE`: Device to use (`auto`, `cuda`, `cpu`, `mps`)
- `LOG_LEVEL`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `PORT`: Server port (default: `8000`)
- `AUDIO_CODEC`: Audio encoding backend (`auto`, `torchcodec`, `ffmpeg`) - see [Audio Codec Configuration](AUDIO_CODEC_CONFIG.md)

> **Note:** For audio format conversion configuration, see the [Audio Codec Configuration Guide](AUDIO_CODEC_CONFIG.md) which explains how to configure MP3/Opus/AAC encoding for CPU and GPU deployments.

### Zero-Shot Voice Cloning
Generate audio that sounds similar as the provided [reference audio](./examples/voice_prompts/belinda.wav).

```bash
python3 examples/generation.py \
--transcript "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years." \
--ref_audio belinda \
--temperature 0.3 \
--out_path generation.wav
```

The generation script will automatically use `cuda:0` if it founds cuda is available. To change the device id, specify `--device_id`:

```bash
python3 examples/generation.py \
--transcript "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years." \
--ref_audio belinda \
--temperature 0.3 \
--device_id 0 \
--out_path generation.wav
```

You can also try other voices. Check more example voices in [examples/voice_prompts](./examples/voice_prompts). You can also add your own voice to the folder.

```bash
python3 examples/generation.py \
--transcript "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years." \
--ref_audio broom_salesman \
--temperature 0.3 \
--out_path generation.wav
```

### Single-speaker Generation with Smart Voice
If you do not specify reference voice, the model will decide the voice based on the transcript it sees.

```bash
python3 examples/generation.py \
--transcript "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years." \
--temperature 0.3 \
--out_path generation.wav
```


### Multi-speaker Dialog with Smart Voice
Generate multi-speaker dialog. The model will decide the voices based on the transcript it sees.

```bash
python3 examples/generation.py \
--transcript examples/transcript/multi_speaker/en_argument.txt \
--seed 12345 \
--out_path generation.wav
```

### Multi-speaker Dialog with Voice Clone

Generate multi-speaker dialog with the voices you picked.

```bash
python3 examples/generation.py \
--transcript examples/transcript/multi_speaker/en_argument.txt \
--ref_audio belinda,broom_salesman \
--ref_audio_in_system_message \
--chunk_method speaker \
--seed 12345 \
--out_path generation.wav
```


## Technical Details
<img src="figures/higgs_audio_v2_architecture_combined.png" width=900>


Higgs Audio v2 adopts the "generation variant" depicted in the architecture figure above. Its strong performance is driven by three key technical innovations:
- We developed an automated annotation pipeline that leverages multiple ASR models, sound event classification models, and our in-house audio understanding model. Using this pipeline, we cleaned and annotated 10 million hours audio data, which we refer to as **AudioVerse**. The in-house understanding model is finetuned on top of [Higgs Audio v1 Understanding](https://www.boson.ai/blog/higgs-audio), which adopts the "understanding variant" shown in the architecture figure.
- We trained a unified audio tokenizer from scratch that captures both semantic and acoustic features. We also open-sourced our evaluation set on [HuggingFace](https://huggingface.co/datasets/bosonai/AudioTokenBench). Learn more in the [tokenizer blog](./tech_blogs/TOKENIZER_BLOG.md).
- We proposed the DualFFN architecture, which enhances the LLM’s ability to model acoustics tokens with minimal computational overhead. See the [architecture blog](./tech_blogs/ARCHITECTURE_BLOG.md).

## Evaluation

Here's the performance of Higgs Audio v2 on four benchmarks,  [Seed-TTS Eval](https://github.com/BytedanceSpeech/seed-tts-eval), [Emotional Speech Dataset (ESD)](https://paperswithcode.com/dataset/esd), [EmergentTTS-Eval](https://arxiv.org/abs/2505.23009), and Multi-speaker Eval:

#### Seed-TTS Eval & ESD

We prompt Higgs Audio v2 with the reference text, reference audio, and target text for zero-shot TTS. We use the standard evaluation metrics from Seed-TTS Eval and ESD.

|                              | SeedTTS-Eval| | ESD   |                 |
|------------------------------|--------|--------|---------|-------------------|
|                              | WER ↓ | SIM ↑ | WER ↓ | SIM (emo2vec) ↑ |
| Cosyvoice2                   | 2.28   | 65.49  | 2.71    | 80.48             |
| Qwen2.5-omni†                | 2.33   | 64.10  | -       | -                 |
| ElevenLabs Multilingual V2   | **1.43**   | 50.00  | 1.66    | 65.87             |
| Higgs Audio v1                | 2.18   | 66.27  | **1.49**    | 82.84             |
| Higgs Audio v2 (base)         | 2.44   | **67.70**  | 1.78    | **86.13**         |


#### EmergentTTS-Eval ("Emotions" and "Questions")

Following the [EmergentTTS-Eval Paper](https://arxiv.org/abs/2505.23009), we report the win-rate over "gpt-4o-mini-tts" with the "alloy" voice. The judge model is Gemini 2.5 Pro.

| Model                              | Emotions (%) ↑ | Questions (%) ↑ |
|------------------------------------|--------------|----------------|
| Higgs Audio v2 (base)               | **75.71%**   | **55.71%**         |
| [gpt-4o-audio-preview†](https://platform.openai.com/docs/models/gpt-4o-audio-preview)       | 61.64%       | 47.85%         |
| [Hume.AI](https://www.hume.ai/research)                            | 61.60%       | 43.21%         |
| **BASELINE:** [gpt-4o-mini-tts](https://platform.openai.com/docs/models/gpt-4o-mini-tts)  | 50.00%       | 50.00%         |
| [Qwen 2.5 Omni†](https://github.com/QwenLM/Qwen2.5-Omni)      | 41.60%       | 51.78%         |
| [minimax/speech-02-hd](https://replicate.com/minimax/speech-02-hd)               | 40.86%        | 47.32%         |
| [ElevenLabs Multilingual v2](https://elevenlabs.io/blog/eleven-multilingual-v2)         | 30.35%       | 39.46%         |
| [DeepGram Aura-2](https://deepgram.com/learn/introducing-aura-2-enterprise-text-to-speech)                    | 29.28%       | 48.21%         |
| [Sesame csm-1B](https://github.com/SesameAILabs/csm)                      | 15.96%       | 31.78%         |

<sup><sub>'†' means using the strong-prompting method described in the paper.</sub></sup>


#### Multi-speaker Eval

We also designed a multi-speaker evaluation benchmark to evaluate the capability of Higgs Audio v2 for multi-speaker dialog generation. The benchmark contains three subsets

- `two-speaker-conversation`: 1000 synthetic dialogues involving two speakers. We fix two reference audio clips to evaluate the model's ability in double voice cloning for utterances ranging from 4 to 10 dialogues between two randomly chosen persona.
- `small talk (no ref)`: 250 synthetic dialogues curated in the same way as above, but are characterized by short utterances and a limited number of turns (4–6), we do not fix reference audios in this case and this set is designed to evaluate the model's ability to automatically assign appropriate voices to speakers.
- `small talk (ref)`: 250 synthetic dialogues similar to above, but contains even shorter utterances as this set is meant to include reference clips in it's context, similar to `two-speaker-conversation`.


We report the word-error-rate (WER) and the geometric mean between intra-speaker similarity and inter-speaker dis-similarity on these three subsets. Other than Higgs Audio v2, we also evaluated [MoonCast](https://github.com/jzq2000/MoonCast) and [nari-labs/Dia-1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626), two of the most popular open-source models capable of multi-speaker dialog generation. Results are summarized in the following table. We are not able to run [nari-labs/Dia-1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626) on our "two-speaker-conversation" subset due to its strict limitation on the length of the utterances and output audio.

|                                                | two-speaker-conversation |                |small talk |                | small talk (no ref) |                |
| ---------------------------------------------- | -------------- | ------------------ | ---------- | -------------- | ------------------- | -------------- |
|                                                | WER ↓                      | Mean Sim & Dis-sim ↑ | WER ↓       |  Mean Sim & Dis-sim ↑ | WER ↓               | Mean Sim & Dis-sim ↑ |
| [MoonCast](https://github.com/jzq2000/MoonCast) | 38.77                    | 46.02         | **8.33**       | 63.68          | 24.65               | 53.94 |
| [nari-labs/Dia-1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626)         | \-                       | \-             | 17.62      | 63.15          | 19.46               | **61.14**          |
| Higgs Audio v2 (base)     | **18.88**                    | **51.95**          | 11.89      | **67.92**              | **14.65**               | 55.28              |

## Contribution and Support

For contribution and support guidelines, please see the support guidelines at [SUPPORT_GUIDELINES.md](SUPPORT_GUIDELINES.md).

## Citation

If you feel the repository is helpful, please kindly cite as:

```
@misc{higgsaudio2025,
  author       = {{Boson AI}},
  title        = {{Higgs Audio V2: Redefining Expressiveness in Audio Generation}},
  year         = {2025},
  howpublished = {\url{https://github.com/boson-ai/higgs-audio}},
  note         = {GitHub repository. Release blog available at \url{https://www.boson.ai/blog/higgs-audio-v2}},
}
```

## Third-Party Licenses

The `boson_multimodal/audio_processing/` directory contains code derived from third-party repositories, primarily from [xcodec](https://github.com/zhenye234/xcodec). Please see the [`LICENSE`](boson_multimodal/audio_processing/LICENSE) in that directory for complete attribution and licensing information.

## Credits & Acknowledgements

### OpenAI-Compatible API Server

The OpenAI-compatible TTS API server implementation was inspired by and adapted from the architecture described in [Kokoro-FastAPI](https://github.com/groxaxo/Kokoro-FastAPI). We thank the Kokoro-FastAPI project for their excellent documentation and design patterns, particularly:

- FastAPI endpoint structure and routing patterns
- Text sanitization and normalization strategies  
- Request/response schema designs
- Streaming audio handling approaches

**Note**: This implementation does not imply endorsement by the Kokoro-FastAPI project or Boson AI. The API server is an independent implementation that uses Higgs Audio models.

### Higgs Audio Model

The core TTS capabilities are powered by **Higgs Audio v2.5** from Boson AI:

- **Model**: [bosonai/higgs-audio-v2-generation-3B-base](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base)
- **Tokenizer**: [bosonai/higgs-audio-v2-tokenizer](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer)
- **Citation**: See [Citation](#citation) section above

For more information about Higgs Audio, visit:
- [Higgs Audio V2.5 Blog Post](https://www.boson.ai/blog/higgs-audio-v2.5)
- [Higgs Audio V2 Blog Post](https://www.boson.ai/blog/higgs-audio-v2)
- [Original Repository](https://github.com/boson-ai/higgs-audio)

## We Are Hiring!

If you are passionate about multimodal AI, speech/audio models, or large-scale systems, 
check out our open positions at [Boson AI Careers](https://jobs.lever.co/bosonai).
