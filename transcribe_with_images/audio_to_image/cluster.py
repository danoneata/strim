import pdb
import pickle
import random

import click
import numpy as np
import h5py

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from transcribe_with_images.train import (
    Flickr8kDataset,
    H5_PATH_IMAGE,
    load_image_feat,
)


SEED = 1337
random.seed(SEED)
np.random.seed(SEED)


DATASET_NAME = "flickr8k"
AUDIO_MODEL_NAME = "wav2vec2-xls-r-2b"
IMAGE_MODEL_NAME = "blip-base"
SPLIT = "train"


def get_path_pca(num_pca_dimensions):
    return "output/kmeans-image-feat/pca-{}-{}-d-{}.pkl".format(
        DATASET_NAME,
        IMAGE_MODEL_NAME,
        num_pca_dimensions,
    )
    

def get_path_kmeans(num_clusters, num_pca_dimensions=None):
    suffix = "-d-{}".format(num_pca_dimensions) if num_pca_dimensions is not None else ""
    return "output/kmeans-image-feat/kmeans-{}-{}-k-{}{}.pkl".format(
        DATASET_NAME,
        IMAGE_MODEL_NAME,
        num_clusters,
        suffix,
    )


@click.command()
@click.option("-k", "num_clusters", type=click.INT)
@click.option("-d", "num_pca_dimensions", type=click.INT)
def main(num_clusters, num_pca_dimensions):
    h5_path = H5_PATH_IMAGE.format(IMAGE_MODEL_NAME, DATASET_NAME, SPLIT)
    dataset = Flickr8kDataset(split=SPLIT)

    num_feats_per_image = 32
    num_images = 5_000

    def load_features_subset(image_h5, i):
        feat = load_image_feat(image_h5, dataset[i]).squeeze(0)
        idxs = np.random.choice(feat.shape[0], num_feats_per_image, replace=False)
        return feat[idxs]

    selected_images = np.random.choice(len(dataset), num_images, replace=False)

    with h5py.File(h5_path, "r") as image_h5:
        features = [load_features_subset(image_h5, i) for i in selected_images]
        features = np.vstack(features)

    if num_pca_dimensions is not None:
        pca = PCA(n_components=num_pca_dimensions)
        features = pca.fit_transform(features)

        with open(get_path_pca(num_pca_dimensions), "wb") as f:
            pickle.dump(pca, f)

    kmeans = KMeans(n_clusters=num_clusters, n_init="auto", verbose=1)
    kmeans.fit(features)

    with open(get_path_kmeans(num_clusters, num_pca_dimensions), "wb") as f:
        pickle.dump(kmeans, f)


if __name__ == "__main__":
    main()
