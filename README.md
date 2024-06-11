# Strim: Speech translation with images

This repository contains the code corresponding to our paper:

> Dan Oneata and Herman Kamper.
> Translating speech with just images.
> Interspeech, 2024.

The main idea of the paper is to distil the knowledge of an image captioning system into a speech paraphrasing system.
When the audio is in a foreign language the system will perform speech translation, outputting an English paraphrase.
This process is illustrated schematically in the figure below:

<img width="400" src="vignette.png"></img>

# Setup

The code depends on PyTorch, which we recommend installing with conda; for example:

```bash
conda create -n strim python=3.12
conda activate strim
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Then you can install this code as a library with:

```bash
pip install -e .
```

# Experiments

The main steps are the following:

1. Extract image features and generate image captions:
```bash
for s in train dev test; do
    python strim/scripts/run_image_captioner.py -m blip-base -d flickr8k --split $s
done
```
2. Extract audio features:
```bash
for s in train dev test; do
    python strim/scripts/extract_audio_features.py --split $s
done
```
3. Learn a mapping network from audio features to image features:
```bash
python strim/audio_to_text/cross_attention/train.py -c tiny
# or use accelerate for multi-GPU training
accelerate launch strim/audio_to_text/cross_attention/train.py -c 00-blip2-diverse
```

# Miscellany

Apart from the cross-attention-based audio-to-text network (in `strim/audio_to_text/cross_attention`),
the repository implements two other approaches that were briefly discussed in the paper, but which gave worse results:

- **Prompt tuning** (in `strim/audio_to_text/prompt_tuning`) maps the audio to a soft prompt to guide the decoding towards translation.
- **Audio to image mapper** (in `strim/audio_to_image`) maps the audio to image features (instead of text) and use those as input to a frozen pretrained image captioning system.
