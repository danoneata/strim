import json

import numpy as np
import pandas as pd

from sacrebleu.metrics import BLEU

from transcribe_with_images.train import Flickr8kDataset


def load_results(num_clusters):
    dataset_name = "flickr8k"
    image_model_name = "blip-base"
    filename = "{}-{}-{}".format(dataset_name, image_model_name, num_clusters)
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


def get_bleu_vs_gt(results):
    bleu = BLEU()
    dataset = Flickr8kDataset(split="train")
    key_to_captions = dataset.get_image_key_to_captions()
    sys = [result["generated-caption-reconstructed"] for result in results]
    refs = [key_to_captions[result["key-image"]] for result in results]
    return bleu.corpus_score(sys, refs).score


def get_evaluation_suite(num_clusters):
    results = load_results(num_clusters)
    return {
        "num_clusters": num_clusters,
        "mse": get_mse(results),
        "bleu": get_bleu(results),
        "bleu-vs-gt": get_bleu_vs_gt(results),
    }


def main():
    # num_clusters = [64, 128, 256, 512]
    num_clusters = [64, 128]
    scores = [get_evaluation_suite(k) for k in num_clusters]
    df = pd.DataFrame(scores)
    print(df)


if __name__ == "__main__":
    main()