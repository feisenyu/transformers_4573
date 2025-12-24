<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers_4573-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers_4573-logo-light.svg">
    <img alt="Hugging Face transformers_4573 Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers_4573-logo-light.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
    <a href="https://huggingface.com/models"><img alt="Checkpoints on Hub" src="https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen"></a>
    <a href="https://circleci.com/gh/huggingface/transformers_4573"><img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers_4573/main"></a>
    <a href="https://github.com/huggingface/transformers_4573/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers_4573.svg?color=blue"></a>
    <a href="https://huggingface.co/docs/transformers_4573/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers_4573/index.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/huggingface/transformers_4573/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers_4573.svg"></a>
    <a href="https://github.com/huggingface/transformers_4573/blob/main/CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg"></a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/huggingface/transformers_4573/blob/main/i18n/README_zh-hans.md">简体中文</a> |
        <a href="https://github.com/huggingface/transformers_4573/blob/main/i18n/README_zh-hant.md">繁體中文</a> |
        <a href="https://github.com/huggingface/transformers_4573/blob/main/i18n/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/transformers_4573/blob/main/i18n/README_es.md">Español</a> |
        <a href="https://github.com/huggingface/transformers_4573/blob/main/i18n/README_ja.md">日本語</a> |
        <a href="https://github.com/huggingface/transformers_4573/blob/main/i18n/README_hd.md">हिन्दी</a> |
        <a href="https://github.com/huggingface/transformers_4573/blob/main/i18n/README_ru.md">Русский</a> |
        <a href="https://github.com/huggingface/transformers_4573/blob/main/i18n/README_pt-br.md">Português</a> |
        <a href="https://github.com/huggingface/transformers_4573/blob/main/i18n/README_te.md">తెలుగు</a> |
        <a href="https://github.com/huggingface/transformers_4573/blob/main/i18n/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/transformers_4573/blob/main/i18n/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/transformers_4573/blob/main/i18n/README_it.md">Italiano</a> |
        <a href="https://github.com/huggingface/transformers_4573/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers_4573/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers_4573/blob/main/i18n/README_ur.md">اردو</a> |
        <a href="https://github.com/huggingface/transformers_4573/blob/main/i18n/README_bn.md">বাংলা</a> |
    </p>
</h4>

<h3 align="center">
    <p>State-of-the-art pretrained models for inference and training</p>
</h3>

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers_4573/transformers_4573_as_a_model_definition.png"/>
</h3>

transformers_4573 acts as the model-definition framework for state-of-the-art machine learning with text, computer
vision, audio, video, and multimodal models, for both inference and training.

It centralizes the model definition so that this definition is agreed upon across the ecosystem. `transformers_4573` is the
pivot across frameworks: if a model definition is supported, it will be compatible with the majority of training
frameworks (Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, ...), inference engines (vLLM, SGLang, TGI, ...),
and adjacent modeling libraries (llama.cpp, mlx, ...) which leverage the model definition from `transformers_4573`.

We pledge to help support new state-of-the-art models and democratize their usage by having their model definition be
simple, customizable, and efficient.

There are over 1M+ transformers_4573 [model checkpoints](https://huggingface.co/models?library=transformers_4573&sort=trending) on the [Hugging Face Hub](https://huggingface.com/models) you can use.

Explore the [Hub](https://huggingface.com/) today to find a model and use transformers_4573 to help you get started right away.

## Installation

transformers_4573 works with Python 3.9+, and [PyTorch](https://pytorch.org/get-started/locally/) 2.1+.

Create and activate a virtual environment with [venv](https://docs.python.org/3/library/venv.html) or [uv](https://docs.astral.sh/uv/), a fast Rust-based Python package and project manager.

```py
# venv
python -m venv .my-env
source .my-env/bin/activate
# uv
uv venv .my-env
source .my-env/bin/activate
```

Install transformers_4573 in your virtual environment.

```py
# pip
pip install "transformers_4573[torch]"

# uv
uv pip install "transformers_4573[torch]"
```

Install transformers_4573 from source if you want the latest changes in the library or are interested in contributing. However, the *latest* version may not be stable. Feel free to open an [issue](https://github.com/huggingface/transformers_4573/issues) if you encounter an error.

```shell
git clone https://github.com/huggingface/transformers_4573.git
cd transformers_4573

# pip
pip install '.[torch]'

# uv
uv pip install '.[torch]'
```

## Quickstart

Get started with transformers_4573 right away with the [Pipeline](https://huggingface.co/docs/transformers_4573/pipeline_tutorial) API. The `Pipeline` is a high-level inference class that supports text, audio, vision, and multimodal tasks. It handles preprocessing the input and returns the appropriate output.

Instantiate a pipeline and specify model to use for text generation. The model is downloaded and cached so you can easily reuse it again. Finally, pass some text to prompt the model.

```py
from transformers_4573 import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
[{'generated_text': 'the secret to baking a really good cake is 1) to use the right ingredients and 2) to follow the recipe exactly. the recipe for the cake is as follows: 1 cup of sugar, 1 cup of flour, 1 cup of milk, 1 cup of butter, 1 cup of eggs, 1 cup of chocolate chips. if you want to make 2 cakes, how much sugar do you need? To make 2 cakes, you will need 2 cups of sugar.'}]
```

To chat with a model, the usage pattern is the same. The only difference is you need to construct a chat history (the input to `Pipeline`) between you and the system.

> [!TIP]
> You can also chat with a model directly from the command line, as long as [`transformers_4573 serve` is running](https://huggingface.co/docs/transformers_4573/main/en/serving).
> ```shell
> transformers_4573 chat Qwen/Qwen2.5-0.5B-Instruct
> ```

```py
import torch
from transformers_4573 import pipeline

chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]

pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

Expand the examples below to see how `Pipeline` works for different modalities and tasks.

<details>
<summary>Automatic speech recognition</summary>

```py
from transformers_4573 import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

</details>

<details>
<summary>Image classification</summary>

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"></a>
</h3>

```py
from transformers_4573 import pipeline

pipeline = pipeline(task="image-classification", model="facebook/dinov2-small-imagenet1k-1-layer")
pipeline("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
[{'label': 'macaw', 'score': 0.997848391532898},
 {'label': 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita',
  'score': 0.0016551691805943847},
 {'label': 'lorikeet', 'score': 0.00018523589824326336},
 {'label': 'African grey, African gray, Psittacus erithacus',
  'score': 7.85409429227002e-05},
 {'label': 'quail', 'score': 5.502637941390276e-05}]
```

</details>

<details>
<summary>Visual question answering</summary>

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers_4573/tasks/idefics-few-shot.jpg"></a>
</h3>

```py
from transformers_4573 import pipeline

pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
pipeline(
    image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers_4573/tasks/idefics-few-shot.jpg",
    question="What is in the image?",
)
[{'answer': 'statue of liberty'}]
```

</details>

## Why should I use transformers_4573?

1. Easy-to-use state-of-the-art models:
    - High performance on natural language understanding & generation, computer vision, audio, video, and multimodal tasks.
    - Low barrier to entry for researchers, engineers, and developers.
    - Few user-facing abstractions with just three classes to learn.
    - A unified API for using all our pretrained models.

1. Lower compute costs, smaller carbon footprint:
    - Share trained models instead of training from scratch.
    - Reduce compute time and production costs.
    - Dozens of model architectures with 1M+ pretrained checkpoints across all modalities.

1. Choose the right framework for every part of a models lifetime:
    - Train state-of-the-art models in 3 lines of code.
    - Move a single model between PyTorch/JAX/TF2.0 frameworks at will.
    - Pick the right framework for training, evaluation, and production.

1. Easily customize a model or an example to your needs:
    - We provide examples for each architecture to reproduce the results published by its original authors.
    - Model internals are exposed as consistently as possible.
    - Model files can be used independently of the library for quick experiments.

<a target="_blank" href="https://huggingface.co/enterprise">
    <img alt="Hugging Face Enterprise Hub" src="https://github.com/user-attachments/assets/247fb16d-d251-4583-96c4-d3d76dda4925">
</a><br>

## Why shouldn't I use transformers_4573?

- This library is not a modular toolbox of building blocks for neural nets. The code in the model files is not refactored with additional abstractions on purpose, so that researchers can quickly iterate on each of the models without diving into additional abstractions/files.
- The training API is optimized to work with PyTorch models provided by transformers_4573. For generic machine learning loops, you should use another library like [Accelerate](https://huggingface.co/docs/accelerate).
- The [example scripts](https://github.com/huggingface/transformers_4573/tree/main/examples) are only *examples*. They may not necessarily work out-of-the-box on your specific use case and you'll need to adapt the code for it to work.

## 100 projects using transformers_4573

transformers_4573 is more than a toolkit to use pretrained models, it's a community of projects built around it and the
Hugging Face Hub. We want transformers_4573 to enable developers, researchers, students, professors, engineers, and anyone
else to build their dream projects.

In order to celebrate transformers_4573 100,000 stars, we wanted to put the spotlight on the
community with the [awesome-transformers_4573](./awesome-transformers_4573.md) page which lists 100
incredible projects built with transformers_4573.

If you own or use a project that you believe should be part of the list, please open a PR to add it!

## Example models

You can test most of our models directly on their [Hub model pages](https://huggingface.co/models).

Expand each modality below to see a few example models for various use cases.

<details>
<summary>Audio</summary>

- Audio classification with [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo)
- Automatic speech recognition with [Moonshine](https://huggingface.co/UsefulSensors/moonshine)
- Keyword spotting with [Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks)
- Speech to speech generation with [Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16)
- Text to audio with [MusicGen](https://huggingface.co/facebook/musicgen-large)
- Text to speech with [Bark](https://huggingface.co/suno/bark)

</details>

<details>
<summary>Computer vision</summary>

- Automatic mask generation with [SAM](https://huggingface.co/facebook/sam-vit-base)
- Depth estimation with [DepthPro](https://huggingface.co/apple/DepthPro-hf)
- Image classification with [DINO v2](https://huggingface.co/facebook/dinov2-base)
- Keypoint detection with [SuperPoint](https://huggingface.co/magic-leap-community/superpoint)
- Keypoint matching with [SuperGlue](https://huggingface.co/magic-leap-community/superglue_outdoor)
- Object detection with [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd)
- Pose Estimation with [VitPose](https://huggingface.co/usyd-community/vitpose-base-simple)
- Universal segmentation with [OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large)
- Video classification with [VideoMAE](https://huggingface.co/MCG-NJU/videomae-large)

</details>

<details>
<summary>Multimodal</summary>

- Audio or text to text with [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B)
- Document question answering with [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base)
- Image or text to text with [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- Image captioning [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b)
- OCR-based document understanding with [GOT-OCR2](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)
- Table question answering with [TAPAS](https://huggingface.co/google/tapas-base)
- Unified multimodal understanding and generation with [Emu3](https://huggingface.co/BAAI/Emu3-Gen)
- Vision to text with [Llava-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)
- Visual question answering with [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- Visual referring expression segmentation with [Kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224)

</details>

<details>
<summary>NLP</summary>

- Masked word completion with [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)
- Named entity recognition with [Gemma](https://huggingface.co/google/gemma-2-2b)
- Question answering with [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
- Summarization with [BART](https://huggingface.co/facebook/bart-large-cnn)
- Translation with [T5](https://huggingface.co/google-t5/t5-base)
- Text generation with [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B)
- Text classification with [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B)

</details>

## Citation

We now have a [paper](https://www.aclweb.org/anthology/2020.emnlp-demos.6/) you can cite for the 🤗 transformers_4573 library:
```bibtex
@inproceedings{wolf-etal-2020-transformers_4573,
    title = "transformers_4573: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```
