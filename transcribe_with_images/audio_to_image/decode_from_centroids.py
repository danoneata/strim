import json
import pdb
import pickle
import random

import numpy as np

import click
import h5py
import random
import torch

from tqdm import tqdm

from transformers import BlipProcessor, BlipForConditionalGeneration

from transcribe_with_images.train import (
    Flickr8kDataset,
    H5_PATH_AUDIO,
    H5_PATH_IMAGE,
    get_sample,
)
from transcribe_with_images.predict import generate_caption


random.seed(1337)


@click.command()
@click.option("-i", "--image-model", "image_model_name", default="blip-base")
@click.option("-a", "--audio-model", "audio_model_name", default="wav2vec2-xls-r-2b")
@click.option("-d", "--dataset", "dataset_name", default="flickr8k")
@click.option("-k", "num_clusters", type=click.INT)
@click.option("-v", "--verbose", is_flag=True, default=False)
def main(image_model_name, audio_model_name, dataset_name, num_clusters, verbose):
    device = "cuda"
    split = "train"

    dataset = Flickr8kDataset(split=split)

    audio_h5_path = H5_PATH_AUDIO.format(audio_model_name, dataset_name, split)
    audio_h5 = h5py.File(audio_h5_path, "r")

    image_h5_path = H5_PATH_IMAGE.format(image_model_name, dataset_name, split)
    image_h5 = h5py.File(image_h5_path, "r")

    assert image_model_name == "blip-base"
    image_model_name_full = "Salesforce/blip-image-captioning-base"
    image_captioning_processor = BlipProcessor.from_pretrained(image_model_name_full)
    image_captioning_model = BlipForConditionalGeneration.from_pretrained(
        image_model_name_full
    )
    image_captioning_model = image_captioning_model.to(device)

    filename = "{}-{}-{}".format(dataset_name, image_model_name, num_clusters)
    kmeans_path = f"output/kmeans-image-feat/{filename}.pkl"
    with open(kmeans_path, "rb") as f:
        kmeans = pickle.load(f)

    def get_closest_centroid(feat, kmeans):
        feat = feat.reshape(1, -1)
        idx = kmeans.predict(feat.numpy())[0]
        # print(idx, end=" ")
        return kmeans.cluster_centers_[idx]

    generate_kwargs = dict(
        num_beams=5,
        # no_repeat_ngram_size=2,
        early_stopping=True,
        max_length=50,
    )

    idxs = list(range(len(dataset)))
    idxs = idxs[::5]
    idxs = random.sample(idxs, 512)

    def do1(i):
        sample = get_sample(dataset, audio_h5, image_h5, i)
        image_feats = sample["image-feat"]
        image_feats_orig = image_feats.to(device)

        image_feats_centroids = [
            get_closest_centroid(f, kmeans) for f in image_feats[0]
        ]
        image_feats_centroids = torch.tensor(np.array(image_feats_centroids))
        image_feats_centroids = image_feats_centroids.to(device)
        image_feats_centroids = image_feats_centroids.unsqueeze(0)

        generated_caption_1 = generate_caption(
            image_captioning_model,
            image_captioning_processor,
            image_feats_orig,
            **generate_kwargs,
        )
        generated_caption_2 = generate_caption(
            image_captioning_model,
            image_captioning_processor,
            image_feats_centroids,
            **generate_kwargs,
        )

        error = torch.nn.functional.mse_loss(image_feats_orig, image_feats_centroids).item()

        if verbose:
            print(generated_caption_1)
            print(generated_caption_2)
            print(error)
            print(dataset[i]["text"])
            print()

        return {
            "key-image": dataset[i]["key-image"],
            "generated-caption-original": generated_caption_1,
            "generated-caption-reconstructed": generated_caption_2,
            "error": error,
        }

    if not verbose:
        idxs = tqdm(idxs)

    results = [do1(i) for i in idxs]
    path = f"output/results/kmeans-image-feat/{filename}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    audio_h5.close()
    image_h5.close()


if __name__ == "__main__":
    main()
