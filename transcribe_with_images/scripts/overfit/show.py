import json
import pdb
import random

import h5py
import numpy as np
import streamlit as st

from transcribe_with_images.data import Flickr8kDataset
from transcribe_with_images.predict import get_neg_loss, get_epoch
from transcribe_with_images.scripts.overfit.predict import NUM_BATCHES, BATCH_SIZE, OUT_DIR


st.set_page_config(layout="wide")

image_model_name = "blip-base"
audio_model_name = "wav2vec2-xls-r-2b"
dataset_name = "flickr8k"
mapping_model_name = "large"

dataset = Flickr8kDataset(split="train")
path_hdf5_image = "output/image-captioner/blip-base-flickr8k-train.h5"

# num_samples = NUM_BATCHES * BATCH_SIZE
idxs = list(range(32))


def get_text(f, path):
    return np.array(f[path]).item().decode()


def load_generated_caption_with_model(epoch):
    name = f"{image_model_name}-{audio_model_name}-{dataset_name}-{mapping_model_name}-{epoch}"
    path = f"output/results/overfit/{name}.json"
    with open(path) as f:
        return json.load(f)


epochs = list(range(1, 6))
generated_captions_with_model = {
    epoch: load_generated_caption_with_model(epoch)
    for epoch in epochs
}

MODELS_DIR = OUT_DIR / f"{mapping_model_name}-{dataset_name}-{audio_model_name}-{image_model_name}"
losses = [
    {
        "loss": -get_neg_loss(p),
        "epoch": get_epoch(p),
    }
    for p in MODELS_DIR.glob("*.pt")
]
losses = sorted(losses, key=lambda x: x["epoch"])

with st.sidebar:
    st.markdown("Loss information:")
    losses



with h5py.File(path_hdf5_image, "r") as f:
    for i in idxs:
        sample = dataset[i]
        path = sample["key-image"] + "/" + "generated-caption"
        generated_caption = get_text(f, path).capitalize()
        generated_captions_with_model_str = "\n".join(
            "{:3d} · {}".format(epoch, generated_captions_with_model[epoch][i]["generated-caption"])
            for epoch in epochs
        )

        st.markdown("`{}` · {}".format(*sample["key-audio"]))
        st.audio(sample["path-audio"])
        st.image(sample["path-image"])
        st.markdown("""
- Audio transcription:
```
{}
```
- Generated caption from image:
```
{}
```
- Generated caption from audio:
```
ep. · caption
{}
```
""".format(sample["text"], generated_caption, generated_captions_with_model_str))
        st.markdown("---")