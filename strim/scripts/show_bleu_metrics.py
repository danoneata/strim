import json
import pdb
import random

import numpy as np
from sacrebleu import BLEU

from strim.data import Flickr8kDataset
from strim.audio_to_text.cross_attention.train import GeneratedCaptionsLoader


random.seed(101)


def remove_punct(s):
    return s.replace(".", "").replace(",", "").replace("?", "").replace("!", "")


def bleu_score_corpus_default(bleu, hyps, refs):
    refs = list(zip(*refs))
    hyps = [remove_punct(hyp) for hyp in hyps]
    refs = [[remove_punct(r) for r in rs] for rs in refs]
    return bleu.corpus_score(hyps, refs).score


def bleu_score_sentences(bleu, hyps, refs):
    return np.mean([bleu.sentence_score(h, rs).score for h, rs in zip(hyps, refs)])


bleu_score_corpus = bleu_score_sentences
bleu_score_corpus = bleu_score_corpus_default


def compute_score_inter_annotators(dataset, num_refs=1):
    assert 1 <= num_refs <= 4

    bleu = BLEU(lowercase=True, effective_order=True)
    image_keys_to_captions = dataset.get_image_key_to_captions()
    caption_groups = list(image_keys_to_captions.values())

    def pick_hyp_and_refs(group):
        random.shuffle(group)
        hyp, *refs = group
        return hyp, refs[:num_refs]

    hyps, refs = zip(*[pick_hyp_and_refs(g) for g in caption_groups])
    return bleu_score_corpus(bleu, hyps, refs)


def compute_score_image_captions(dataset, load_generated_captions, num_refs=1):
    assert 1 <= num_refs <= 5

    bleu = BLEU(lowercase=True, effective_order=True)
    image_keys_to_captions = dataset.get_image_key_to_captions()

    def pick_hyp(key):
        sample = {"key-image": key}
        hyps = load_generated_captions(sample)
        random.shuffle(hyps)
        return hyps[0]

    def pick_refs(key):
        refs = image_keys_to_captions[key]
        random.shuffle(refs)
        return refs[:num_refs]

    hyps = [pick_hyp(key) for key in image_keys_to_captions.keys()]
    refs = [pick_refs(key) for key in image_keys_to_captions.keys()]
    return bleu_score_corpus(bleu, hyps, refs)


def compute_score_predictions(dataset, predictions):
    bleu = BLEU(lowercase=True, effective_order=True)
    num_samples = len(dataset)

    def pick_hyp_and_refs(i):
        sample = dataset[i]
        prediction = predictions[i]
        assert sample["key-audio"] == tuple(prediction["key-audio"])
        hyp = prediction["text-prediction"]
        refs = [sample["text"]]
        # print(hyp, refs)
        return hyp, refs

    idxs = list(range(num_samples))
    idxs = random.sample(idxs, k=num_samples // 5)
    hyps, refs = zip(*[pick_hyp_and_refs(i) for i in idxs])
    return bleu_score_corpus(bleu, hyps, refs)


def main():
    split = "test"
    dataset = Flickr8kDataset(split=split)

    print("# BLEU score for inter-annotators")
    num_repeats = 5
    scores_to_print = []
    for num_refs in range(1, 5):
        scores = [
            compute_score_inter_annotators(dataset, num_refs)
            for _ in range(num_repeats)
        ]
        score_to_print = "{:.2f}±{:.1f}".format(np.mean(scores), 2 * np.std(scores))
        scores_to_print.append(score_to_print)
        print("num. refs: {:d} · BLEU: {}".format(num_refs, score_to_print))
    print(" ".join(scores_to_print))
    print()

    print("# BLEU score for image captions")
    load_generated_captions = GeneratedCaptionsLoader("blip-base", "flickr8k", split)
    num_repeats = 5
    scores_to_print = []
    for num_refs in range(1, 6):
        scores = [
            compute_score_image_captions(dataset, load_generated_captions, num_refs)
            for _ in range(num_repeats)
        ]
        score_to_print = "{:.2f}±{:.1f}".format(np.mean(scores), 2 * np.std(scores))
        scores_to_print.append(score_to_print)
        print("num. refs: {:d} · BLEU: {}".format(num_refs, score_to_print))
    print(" ".join(scores_to_print))
    print()

    print("# BLEU score for predictions")
    num_repeats = 5
    # path_predictions = "output/audio-to-text-mapper/predictions/00-00-best-2023-09-21.json"
    path_predictions = "output/audio-to-text-mapper/predictions/00-00-best-2023-11-11.json"
    with open(path_predictions, "r") as f:
        predictions = json.load(f)
    scores = [compute_score_predictions(dataset, predictions) for _ in range(num_repeats)]
    print("num. refs: {:d} · BLEU: {:.2f}±{:.1f}".format(1, np.mean(scores), 2 * np.std(scores)))
    print()


if __name__ == "__main__":
    main()
