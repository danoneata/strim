import json
import os
import pdb

from pathlib import Path

import click
import h5py
import torch

from transformers import BlipProcessor, BlipForConditionalGeneration

from transcribe_with_images.train import (
    AudioToImageMapper,
    Flickr8kDataset,
    H5_PATH_AUDIO,
    H5_PATH_IMAGE,
    MODELS,
    get_sample,
)
from transcribe_with_images.scripts.overfit.train import (
    BATCH_SIZE,
    NUM_BATCHES,
    OUT_DIR,
)
from transcribe_with_images.predict import (
    generate_caption,
    get_epoch,
)


def get_model_path(image_model_name, audio_model_name, dataset_name, mapping_model_name, epoch):
    folder = OUT_DIR / f"{mapping_model_name}-{dataset_name}-{audio_model_name}-{image_model_name}"
    folder = Path(folder)
    model_paths = [folder / f for f in os.listdir(folder)]
    model_paths = [p for p in model_paths if get_epoch(p) == epoch]
    model_path, = model_paths
    return model_path


@click.command()
@click.option("-i", "--image-model", "image_model_name", default="blip-base")
@click.option("-a", "--audio-model", "audio_model_name", default="wav2vec2-xls-r-2b")
@click.option("-d", "--dataset", "dataset_name", default="flickr8k")
@click.option("-m", "--mapping-model", "mapping_model_name")
@click.option("-e", "--epoch", type=click.INT)
def main(image_model_name, audio_model_name, dataset_name, mapping_model_name, epoch):
    device = "cuda"
    split = "train"

    dataset = Flickr8kDataset(split=split)
    audio_h5 = h5py.File(
        H5_PATH_AUDIO.format(audio_model_name, dataset_name, split), "r"
    )
    image_h5 = h5py.File(
        H5_PATH_IMAGE.format(image_model_name, dataset_name, split), "r"
    )

    model_path = get_model_path(image_model_name, audio_model_name, dataset_name, mapping_model_name, epoch)
    model = AudioToImageMapper(
        dim_audio=1920,
        dim_image=768,
        len_image_seq=577,
        **MODELS[mapping_model_name],
    )
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    assert image_model_name == "blip-base"
    image_model_name_full = "Salesforce/blip-image-captioning-base"
    image_captioning_processor = BlipProcessor.from_pretrained(image_model_name_full)
    image_captioning_model = BlipForConditionalGeneration.from_pretrained(
        image_model_name_full
    )
    image_captioning_model = image_captioning_model.to(device)

    results = []

    # for i in range(NUM_BATCHES * BATCH_SIZE):
    for i in range(32):
        datum = dataset[i]
        sample = get_sample(dataset, audio_h5, image_h5, i)

        audio_feat = sample["audio-feat"]
        audio_feat = audio_feat.to(device)
        audio_feat = audio_feat.unsqueeze(0)

        B, T, _ = audio_feat.shape
        assert B == 1
        padding_mask = torch.full((B, T), fill_value=False)
        padding_mask = padding_mask.to(device)

        input = audio_feat, padding_mask
        image_feat_pred = model(input)

        generated_caption = generate_caption(
            image_captioning_model,
            image_captioning_processor,
            image_feat_pred,
        )
        print(generated_caption)

        result = {
            **datum,
            "generated-caption": generated_caption,
        }

        results.append(result)

    audio_h5.close()
    image_h5.close()

    name = f"{image_model_name}-{audio_model_name}-{dataset_name}-{mapping_model_name}-{epoch}"
    with open(f"output/results/overfit/{name}.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
