import pdb
import pickle

from pathlib import Path

import click
import numpy as np
import h5py

from sklearn.cluster import KMeans

from transcribe_with_images.train import (
    Flickr8kDataset,
    H5_PATH_IMAGE,
    load_image_feat,
)


@click.command()
@click.option("-k", "num_clusters", type=click.INT)
def main(num_clusters):
    dataset_name = "flickr8k"
    image_model_name = "blip-base"
    split = "train"
    h5_path = H5_PATH_IMAGE.format(image_model_name, dataset_name, split)

    dataset = Flickr8kDataset(split=split)

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

    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(features)

    checkpoint_path = "output/kmeans-image-feat/{}-{}-{}.pkl".format(
        dataset_name, image_model_name, num_clusters
    )
    with open(checkpoint_path, "wb") as f:
        pickle.dump(kmeans, f)


if __name__ == "__main__":
    main()
