import json
import pdb
import random

from itertools import groupby
from pathlib import Path

import click
import librosa
import numpy as np
import seaborn as sns
import streamlit as st

from matplotlib import pyplot as plt
from toolz import first

from strim.audio_to_text.cross_attention.predict import (
    CONFIGS,
    CONFIGS_PREDICT,
    DATASETS,
)
from strim.scripts.extract_audio_features import SAMPLING_RATE
from strim.utils import read_file


def fmt_txt(s):
    return s.replace(" .", "").lower()


def fmt_tokens(token):
    if token == "<|endoftext|>":
        return "<eos>"
    else:
        return token.replace("Ġ", "_")


def show_all_small(attentions):
    fig, axs = plt.subplots(nrows=12, ncols=12, figsize=(10, 2.5))
    for i in range(12):
        for j in range(12):
            A = attentions[i, j]
            sns.heatmap(A, ax=axs[i, j], cbar=False)
            axs[i, j].set_xticklabels([])
            axs[i, j].set_yticklabels([])
    return fig.set_tight_layout(True)


def show_all_bipartite(attentions, i, layer_num):
    from chalk import make_path, concat, text, vstrut
    from colour import Color

    path = f"/tmp/att-id-{i}-layer-num-{layer_num}.svg"
    if Path(path).exists():
        return path

    τ = 0.05
    Δ = 0.4
    ε = 0.2

    def show1(i, j):
        A = attentions[i, j]
        y1, y2 = np.where(A > τ)
        lines = [
            make_path([(0, y1 / A.shape[0]), (Δ, y2 / A.shape[1])]).line_width(0.5 * A[y1, y2])
            for y1, y2 in zip(y1, y2)
        ]
        # return text("L{} H{}".format(i, j), size=0.1).fill_color(Color("black")) // vstrut(0.1) // concat(lines)
        return concat(lines)

    dias = [
        show1(layer_num, h).translate(h * Δ * (1 + ε), 0)
        # for l in range(6)
        for h in range(12)
    ]
    dia = concat(dias)
    dia.render_svg(path, 100)

    return path


def load_alignments_en():
    def parse_line(line):
        key, _, start, duration, word = line.strip().split()
        key_img, num = key.split("#")
        key_img = key_img.split(".")[0]
        start = float(start)
        end = start + float(duration)
        num = int(num)
        return (key_img, num), start, end, word
    path = "/home/doneata/work/herman-semantic-flickr/data/flickr_8k.ctm"
    return {
        key: [
            {
                "start": start,
                "end": end,
                "word": word,
            }
            for _, start, end, word in group
        ]
        for key, group in groupby(read_file(path, parse_line), key=first)
    }


@click.command()
@click.option("-c", "--config", "config_name")
@click.option("-p", "--config-predict", "config_predict_name")
def main(config_name, config_predict_name):
    split = "test"

    config = CONFIGS[config_name]
    config_predict = CONFIGS_PREDICT[config_predict_name]

    dataset_name = config_predict["dataset-name"]
    dataset = DATASETS[dataset_name](split=split)

    output_dir = Path(f"output/audio-to-text-mapper/predictions-with-attentions")

    path_predictions = output_dir / f"{config_name}-{config_predict_name}.json"
    with open(path_predictions, "r") as f:
        predictions = json.load(f)

    FEATURES_RESOLUTION = 0.02  # 20ms

    with st.sidebar:
        layer_num = st.number_input("layer number", min_value=0, max_value=11, value=11, step=1)
        head_num = st.number_input("head number", min_value=0, max_value=11, value=0, step=1)

    random.seed(1337)
    idxs = list(range(len(dataset)))
    idxs = random.sample(idxs, k=30)

    alignments_en = load_alignments_en()

    for i in idxs:
        sample = dataset[i]
        preds = predictions[i]
        tokens = preds["text-tokens"]
        tokens = [fmt_tokens(token) for token in tokens]

        audio, _ = librosa.load(sample["path-audio"], mono=True, sr=SAMPLING_RATE)

        path_attentions = output_dir / f"{config_name}-{config_predict_name}-cross-attentions" / f"{i}.npy"
        attentions = np.load(path_attentions, allow_pickle=True)

        # st.write(attentions.max(axis=(2, 3)))
        # A = attentions.mean(axis=(0, 1))
        A = attentions[layer_num, head_num]
        # _, num_frames = A.shape
        # xticks = np.arange(0, num_frames) * FEATURES_RESOLUTION * SAMPLING_RATE

        # print(predictions[i])
        # print(attentions[i])

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
        sns.heatmap(A, ax=axs[0], cbar=False, vmin=0, vmax=1)
        axs[0].set_yticklabels(tokens, rotation=0)
        axs[0].set_xticklabels([])
        axs[0].set_xticks([])

        for alignment in alignments_en[sample["key-audio"]]:
            s = SAMPLING_RATE * alignment["start"]
            e = SAMPLING_RATE * alignment["end"]
            axs[1].axvline(s, color="gray", linestyle="--")
            axs[1].axvline(e, color="gray", linestyle="--")
            axs[1].text(
                (s + e) / 2,
                -0.9,
                alignment["word"],
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=8,
            )
        axs[1].plot(audio)
        axs[1].set_xlim(0, len(audio))
        axs[1].set_ylim(-1, 1)

        fig.tight_layout()

        st.pyplot(fig)
        st.image(show_all_bipartite(attentions, i, layer_num), caption="attentions across all heads")
        st.audio(sample["path-audio"])
        st.markdown("{}".format(fmt_txt(sample["text"])))
        st.markdown("---")


if __name__ == "__main__":
    main()
