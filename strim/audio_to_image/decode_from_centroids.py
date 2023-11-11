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

from sklearn.preprocessing import FunctionTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration

from strim.audio_to_image.train import (
    Flickr8kDataset,
    H5_PATH_AUDIO,
    H5_PATH_IMAGE,
    get_sample,
)
from strim.audio_to_image.predict import generate_caption
from strim.audio_to_image.cluster import (
    get_path_kmeans,
    get_path_pca,
    DATASET_NAME,
    AUDIO_MODEL_NAME,
    IMAGE_MODEL_NAME,
    SPLIT,
)


random.seed(1337)


@click.command()
# @click.option("-i", "--image-model", "image_model_name", default="blip-base")
# @click.option("-a", "--audio-model", "audio_model_name", default="wav2vec2-xls-r-2b")
# @click.option("-d", "--dataset", "dataset_name", default="flickr8k")
@click.option("-k", "num_clusters", type=click.INT)
@click.option("-d", "num_pca_dimensions", type=click.INT)
@click.option("-v", "--verbose", is_flag=True, default=False)
def main(num_clusters, num_pca_dimensions, verbose):
    device = "cuda"
    dataset = Flickr8kDataset(split=SPLIT)

    audio_h5_path = H5_PATH_AUDIO.format(AUDIO_MODEL_NAME, DATASET_NAME, SPLIT)
    audio_h5 = h5py.File(audio_h5_path, "r")

    image_h5_path = H5_PATH_IMAGE.format(IMAGE_MODEL_NAME, DATASET_NAME, SPLIT)
    image_h5 = h5py.File(image_h5_path, "r")

    assert IMAGE_MODEL_NAME == "blip-base"
    image_model_name_full = "Salesforce/blip-image-captioning-base"
    image_captioning_processor = BlipProcessor.from_pretrained(image_model_name_full)
    image_captioning_model = BlipForConditionalGeneration.from_pretrained(
        image_model_name_full
    )
    image_captioning_model = image_captioning_model.to(device)

    if num_pca_dimensions is not None:
        with open(get_path_pca(num_pca_dimensions), "rb") as f:
            pca = pickle.load(f)
    else:
        pca = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)

    with open(get_path_kmeans(num_clusters, num_pca_dimensions), "rb") as f:
        kmeans = pickle.load(f)

    centroids = kmeans.cluster_centers_
    centroids = pca.inverse_transform(centroids)

    def get_closest_centroid(feat, kmeans):
        feat = feat.reshape(1, -1).numpy()
        feat = pca.transform(feat)
        idx = kmeans.predict(feat)[0]
        centroid = centroids[idx]
        return centroid

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
        image_feats_centroids = image_feats_centroids.float()

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

        error = torch.nn.functional.mse_loss(
            image_feats_orig, image_feats_centroids
        ).item()

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

    filename = "{}-{}-k-{}-d-{}".format(
        DATASET_NAME,
        IMAGE_MODEL_NAME,
        num_clusters,
        num_pca_dimensions,
    )
    path = f"output/results/kmeans-image-feat/{filename}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    audio_h5.close()
    image_h5.close()


if __name__ == "__main__":
    main()
