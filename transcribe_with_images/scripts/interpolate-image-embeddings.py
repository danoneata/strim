import os
import pdb

from pathlib import Path

import click
import h5py
import torch

from transformers import BlipProcessor, BlipForConditionalGeneration

from transcribe_with_images.audio_to_image.train import (
    Flickr8kDataset,
    H5_PATH_AUDIO,
    H5_PATH_IMAGE,
    get_sample,
)
from transcribe_with_images.audio_to_image.predict import generate_caption


@click.command()
@click.option("-i", "--image-model", "image_model_name", default="blip-base")
@click.option("-a", "--audio-model", "audio_model_name", default="wav2vec2-xls-r-2b")
@click.option("-d", "--dataset", "dataset_name", default="flickr8k")
def main(image_model_name, audio_model_name, dataset_name):
    device = "cuda"
    split = "train"

    dataset = Flickr8kDataset(split=split)
    audio_h5 = h5py.File(
        H5_PATH_AUDIO.format(audio_model_name, dataset_name, split), "r"
    )
    image_h5 = h5py.File(
        H5_PATH_IMAGE.format(image_model_name, dataset_name, split), "r"
    )

    assert image_model_name == "blip-base"
    image_model_name_full = "Salesforce/blip-image-captioning-base"
    image_captioning_processor = BlipProcessor.from_pretrained(image_model_name_full)
    image_captioning_model = BlipForConditionalGeneration.from_pretrained(
        image_model_name_full
    )
    image_captioning_model = image_captioning_model.to(device)
    indices = list(range(len(dataset)))

    import random

    for _ in range(32):

        i = random.choice(indices)
        j = random.choice(indices)

        sample0 = get_sample(dataset, audio_h5, image_h5, i)
        sample1 = get_sample(dataset, audio_h5, image_h5, j)

        # print(dataset[0])
        # print(dataset[5 * k])

        def interp(x, y, α):
            return α * x + (1 - α) * y

        def get_image_feat(sample0, sample1, α=0.5):
            image_feat0 = sample0["image-feat"].to(device)
            image_feat1 = sample1["image-feat"].to(device)
            return interp(image_feat0, image_feat1, α)

        for α in [0.0, 0.25, 0.5, 0.75, 1.0]:
            generated_caption = generate_caption(
                image_captioning_model,
                image_captioning_processor,
                get_image_feat(sample0, sample1, α),
                num_beams=5,
                # no_repeat_ngram_size=2,
                early_stopping=True,
            )
            print("α = {:.2f} · {}".format(α, generated_caption))
        print()

    audio_h5.close()
    image_h5.close()


if __name__ == "__main__":
    main()
