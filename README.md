Extract visual features and generate image captions:
```bash
for s in train dev test; do
    python transcribe_with_images/scripts/run_image_captioner.py --split $s
done
```

Extract audio features:
```bash
for s in train dev test; do
    python transcribe_with_images/scripts/extract_audio_features.py --split $s
done
```