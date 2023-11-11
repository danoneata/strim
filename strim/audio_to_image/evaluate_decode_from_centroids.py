import json

import numpy as np
import pandas as pd

from sacrebleu.metrics import BLEU

from strim.audio_to_image.train import Flickr8kDataset



def load_results(num_clusters, num_pca_dimensions):
    dataset_name = "flickr8k"
    image_model_name = "blip-base"
    filename = "{}-{}-k-{}-d-{}".format(dataset_name, image_model_name, num_clusters, num_pca_dimensions)
    path = f"output/results/kmeans-image-feat/{filename}.json"
    with open(path, "r") as f:
        return json.load(f)


def get_mse(results):
    return np.mean([result["error"] for result in results])


def get_bleu(results):
    bleu = BLEU()
    sys = [result["generated-caption-reconstructed"] for result in results]
    refs = [[result["generated-caption-original"]] for result in results]
    return bleu.corpus_score(sys, refs).score


def get_bleu_vs_gt_1(key_to_caption_sys):
    bleu = BLEU()
    dataset = Flickr8kDataset(split="train")
    key_to_captions_ref = dataset.get_image_key_to_captions()
    ref = [key_to_captions_ref[key] for key in key_to_caption_sys.keys()]
    sys = list(key_to_caption_sys.values())
    return bleu.corpus_score(sys, ref).score


def get_bleu_vs_gt(results):
    bleu = BLEU()
    dataset = Flickr8kDataset(split="train")
    key_to_captions = dataset.get_image_key_to_captions()
    sys = [result["generated-caption-reconstructed"] for result in results]
    refs = [key_to_captions[result["key-image"]] for result in results]
    return bleu.corpus_score(sys, refs).score


def get_evaluation_suite(num_clusters, num_pca_dimensions):
    results = load_results(num_clusters, num_pca_dimensions)
    return {
        "num-clusters": num_clusters,
        "num-pca-dimensions": num_pca_dimensions,
        "mse": get_mse(results),
        "bleu": get_bleu(results),
        "bleu-vs-gt": get_bleu_vs_gt(results),
    }


def main():
    # results = load_results(64, None)
    # sys = {
    #     result["key-image"]: result["generated-caption-original"]
    #     for result in results
    # }
    # print(get_bleu_vs_gt_1(sys))

    num_pca_dimensions = [64, 128, 256, 512, 768, None]
    # num_clusters = [64, 128, 256, 512]
    num_clusters = [128]
    scores = [get_evaluation_suite(k, d) for k in num_clusters for d in num_pca_dimensions]
    df = pd.DataFrame(scores)
    print(df)


if __name__ == "__main__":
    main()