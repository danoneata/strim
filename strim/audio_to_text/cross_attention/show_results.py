import json
import random
import pdb

import streamlit as st

from sacrebleu import BLEU

from strim.data import DATASETS
from strim.audio_to_text.cross_attention.train import GeneratedCaptionsLoader


st.set_page_config(layout="wide")


TEMPLATE_RESULTS = """
{}
- `true` · {}
- `pred` · {}
---
groundtruth captions:
{}

"""


def fmt_txt(s):
    return s.replace(" .", "").lower()


def bullet_list(items):
    return "\n".join("- " + item for item in items)


def main():
    PRED_TYPE_INFO = {
        "English (transcripts)": {
            "results-path": "output/audio-to-text-mapper/predictions/00-transcripts-00-transcripts-best.json",
            "dataset": "flickr8k",
        },
        "English": {
            "results-path": "output/audio-to-text-mapper/predictions/00-00-best-2023-11-11.json",
            "dataset": "flickr8k",
        },
        "English (git + diverse)": {
            "results-path": "output/audio-to-text-mapper/predictions/flickr8k-git-large-coco-diverse-None.json",
            "dataset": "flickr8k",
        },
        "Yorùbá (transcripts)": {
            "results-path": "output/audio-to-text-mapper/predictions/00-yfacc-transcripts-00-yfacc-transcripts-best.json",
            "dataset": "yfacc",
        },
        "Yorùbá": {
            "results-path": "output/audio-to-text-mapper/predictions/00-yfacc-00-yfacc-best.json",
            "dataset": "yfacc",
        },
        "Yorùbá (git + sample1)": {
            "results-path": "output/audio-to-text-mapper/predictions/yfacc-git-large-coco-sample1-None.json",
            "dataset": "yfacc",
        },
        "Yorùbá (git + diverse)": {
            "results-path": "output/audio-to-text-mapper/predictions/yfacc-git-large-coco-diverse-None.json",
            "dataset": "yfacc",
        },
    }
    PRED_TYPES = list(PRED_TYPE_INFO.keys())
    ORDER_TYPES = ["random", "by BLEU score (decreasing)", "by BLEU score (increasing)"]

    with st.sidebar:
        pred_type = st.selectbox("predictions", PRED_TYPES)
        order_type = st.selectbox("order", ORDER_TYPES)
        num_samples = st.number_input(
            "num. samples",
            min_value=1,
            max_value=300,
            value=30,
            step=10,
        )

    results_path = PRED_TYPE_INFO[pred_type]["results-path"]
    dataset_name = PRED_TYPE_INFO[pred_type]["dataset"]

    split = "test"
    dataset = DATASETS[dataset_name](split=split)
    dataset_en = DATASETS["flickr8k"](split=split)

    with open(results_path, "r") as f:
        predictions = json.load(f)

    image_key_to_captions = dataset_en.get_image_key_to_captions()
    # image_keys = list(image_key_to_captions.keys())

    bleu = BLEU(lowercase=True, effective_order=True)

    def compute_bleu_score(i):
        sample = dataset[i]
        pred_text = fmt_txt(predictions[i]["text-prediction"])
        true_texts = [fmt_txt(c) for c in image_key_to_captions[sample["key-image"]]]
        return bleu.sentence_score(pred_text, true_texts).score

    num_samples_total = len(dataset)
    selected_idxs = list(range(num_samples_total))
    bleu_scores = [compute_bleu_score(i) for i in range(num_samples_total)]

    if order_type == "random":
        random.shuffle(selected_idxs)
    else:
        to_reverse = order_type == "by BLEU score (decreasing)"
        selected_idxs = sorted(
            selected_idxs, key=lambda i: bleu_scores[i], reverse=to_reverse
        )

    selected_idxs = selected_idxs[:num_samples]

    for i in selected_idxs:
        sample = dataset[i]

        if pred_type.startswith("Yorùbá"):
            true_text_yo = "- `true/yo` · " + fmt_txt(sample["text-yo"])
        else:
            true_text_yo = ""

        pred_text = fmt_txt(predictions[i]["text-prediction"])
        true_text = fmt_txt(sample["text"])
        true_texts = [fmt_txt(c) for c in image_key_to_captions[sample["key-image"]]]

        st.markdown(
            "{} · image id: `{}` · caption id: {} ◇ BLUE: {:.1f}%".format(
                i,
                *sample["key-audio"],
                bleu_scores[i],
            )
        )
        col1, col2 = st.columns(2)
        col1.image(str(dataset.get_image_path(sample)))
        col2.audio(str(dataset.get_audio_path(sample)))
        col2.markdown(
            TEMPLATE_RESULTS.format(
                true_text_yo,
                true_text,
                pred_text,
                bullet_list(true_texts),
            )
        )
        st.markdown("---")


if __name__ == "__main__":
    main()
