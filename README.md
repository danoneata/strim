# Strim: Speech transcribing with images

In this project we distil the knowledge of an image captioning system to a speech paraphrasing system.

---

Extract image features and generate image captions:
```bash
for s in train dev test; do
    python strim/scripts/run_image_captioner.py -m blip-base -d flickr8k --split $s
done
```

Extract audio features:
```bash
for s in train dev test; do
    python strim/scripts/extract_audio_features.py --split $s
done
```

Learn a mapping network from audio features to image features:
```bash
python strim/audio_to_image/train.py -m tiny
```
