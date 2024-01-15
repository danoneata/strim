import json
import pdb
import random

from itertools import groupby

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
    image_key_to_captions = dataset.get_image_key_to_captions()
    caption_groups = list(image_key_to_captions.values())

    def pick_hyp_and_refs(group):
        random.shuffle(group)
        hyp, *refs = group
        return hyp, refs[:num_refs]

    hyps, refs = zip(*[pick_hyp_and_refs(g) for g in caption_groups])
    return bleu_score_corpus(bleu, hyps, refs)


def compute_score_image_captions(dataset, load_generated_captions, num_refs=1):
    assert 1 <= num_refs <= 5

    bleu = BLEU(lowercase=True, effective_order=True)
    image_key_to_captions = dataset.get_image_key_to_captions()

    def pick_hyp(key):
        sample = {"key-image": key}
        hyps = load_generated_captions(sample)
        random.shuffle(hyps)
        return hyps[0]

    def pick_refs(key):
        refs = image_key_to_captions[key]
        random.shuffle(refs)
        return refs[:num_refs]

    hyps = [pick_hyp(key) for key in image_key_to_captions.keys()]
    refs = [pick_refs(key) for key in image_key_to_captions.keys()]
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


def compute_score_predictions_lenient(dataset, predictions, num_refs=1):
    assert 1 <= num_refs <= 5

    bleu = BLEU(lowercase=True, effective_order=True)
    image_key_to_captions = dataset.get_image_key_to_captions()
    image_key_to_predictions = {
        key: list(group)
        for key, group in groupby(predictions, key=lambda p: p["key-audio"][0])
    }

    def pick_hyp_and_refs(key):
        preds = image_key_to_predictions[key]
        random.shuffle(preds)
        pred = preds[0]

        i = pred["key-audio"][1]
        hyp = pred["text-prediction"]

        # Select the reference corresponding to the prediction and `num_refs - 1` others.
        refs_all = image_key_to_captions[key]
        ref_corr = refs_all[i]
        refs_rest = [r for j, r in enumerate(refs_all) if j != i]
        refs = [ref_corr] + random.sample(refs_rest, k=num_refs - 1)

        # if num_refs == 5:
        #     assert set(refs) == set(refs_all)

        return hyp, refs

    image_keys_c = image_key_to_captions.keys()
    image_keys_p = image_key_to_predictions.keys()

    if set(image_keys_c) != set(image_keys_p):
        print("WARN Image keys between captions and prediction do not match.")
        print(".... Using the predictions keys.")

    image_keys = image_keys_p

    hyps_and_refs = [pick_hyp_and_refs(key) for key in image_keys]
    hyps, refs = zip(*hyps_and_refs)
    return bleu_score_corpus(bleu, hyps, refs)


def main():
    split = "test"
    dataset = Flickr8kDataset(split=split)

    # print("# BLEU score for true image captions (inter-annotators)")
    # num_repeats = 5
    # scores_to_print = []
    # for num_refs in range(1, 5):
    #     scores = [
    #         compute_score_inter_annotators(dataset, num_refs)
    #         for _ in range(num_repeats)
    #     ]
    #     score_to_print = "{:.2f}±{:.1f}".format(np.mean(scores), 2 * np.std(scores))
    #     scores_to_print.append(score_to_print)
    #     print("num. refs: {:d} · BLEU: {}".format(num_refs, score_to_print))
    # print(" ".join(scores_to_print))
    # print()

    # def evaluate_generated_image_captions(model_name):
    #     print("# BLEU score for generated image captions: " + model_name)
    #     load_generated_captions = GeneratedCaptionsLoader(model_name, "flickr8k", split)
    #     num_repeats = 5
    #     scores_to_print = []
    #     for num_refs in range(1, 6):
    #         scores = [
    #             compute_score_image_captions(dataset, load_generated_captions, num_refs)
    #             for _ in range(num_repeats)
    #         ]
    #         score_to_print = "{:.2f}±{:.1f}".format(np.mean(scores), 2 * np.std(scores))
    #         scores_to_print.append(score_to_print)
    #         print("num. refs: {:d} · BLEU: {}".format(num_refs, score_to_print))
    #     print(" ".join(scores_to_print))
    #     print()

    # evaluate_generated_image_captions("blip-base")
    # evaluate_generated_image_captions("blip-large")

    # print("# BLEU score for predictions")
    # num_repeats = 5
    # # path_predictions = "output/audio-to-text-mapper/predictions/00-00-best-2023-09-21.json"
    # path_predictions = "output/audio-to-text-mapper/predictions/00-00-best-2023-11-11.json"
    # with open(path_predictions, "r") as f:
    #     predictions = json.load(f)
    # scores = [compute_score_predictions(dataset, predictions) for _ in range(num_repeats)]
    # print("num. refs: {:d} · BLEU: {:.2f}±{:.1f}".format(1, np.mean(scores), 2 * np.std(scores)))
    # print()

    def eval_predictions(path):
        num_repeats = 5

        with open(path, "r") as f:
            predictions = json.load(f)

        scores_to_print = []
        for num_refs in range(1, 6):
            scores = [
                compute_score_predictions_lenient(dataset, predictions, num_refs)
                for _ in range(num_repeats)
            ]
            score_to_print = "{:.2f}±{:.1f}".format(np.mean(scores), 2 * np.std(scores))
            scores_to_print.append(score_to_print)
            print("num. refs: {:d} · BLEU: {}".format(num_refs, score_to_print))

        print(" ".join(scores_to_print))
        print()

    # print("# BLEU score for English predictions (lenient)")
    # path_predictions = "output/audio-to-text-mapper/predictions/00-00-best-2023-11-11.json"
    # path_predictions = "output/audio-to-text-mapper/predictions/00-transcripts-00-transcripts-best.json"
    # eval_predictions(path_predictions)

    print("# BLEU score for Yorùbá predictions (lenient)")
    # path_predictions = "output/audio-to-text-mapper/predictions/00-yfacc-00-yfacc-best.json"
    path_predictions = "output/audio-to-text-mapper/predictions/00-yfacc-transcripts-00-yfacc-transcripts-best.json"
    eval_predictions(path_predictions)


if __name__ == "__main__":
    main()
