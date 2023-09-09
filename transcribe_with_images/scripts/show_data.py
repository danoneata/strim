import pdb
import random

import h5py
import numpy as np
import streamlit as st

from transcribe_with_images.data import Flickr8kDataset

dataset = Flickr8kDataset(split="test")
path_hdf5_image = "output/image-captioner/blip-base-flickr8k-test.h5"
path_hdf5_audio = "output/audio-features/wav2vec2-xls-r-2b-flickr8k-test.h5"

idxs = list(range(len(dataset)))
random.shuffle(idxs)
idxs = idxs[:10]


def get_text(f, path):
    return np.array(f[path]).item().decode()


with h5py.File(path_hdf5_image, "r") as f:
    with h5py.File(path_hdf5_audio, "r") as g:
        for i in idxs:
            sample = dataset[i]
            path = sample["key-image"] + "/" + "generated-caption"
            generated_caption = get_text(f, path).capitalize()
            # st.write(sample)
            st.markdown("`{}` · {}".format(*sample["key-audio"]))
            st.audio(sample["path-audio"])
            st.image(sample["path-image"])
            st.markdown("""
- Original caption: {}
- Generated caption: {}
    """.format(sample["text"], generated_caption))
            st.markdown("---")
            # key, i = sample["key-audio"]
            # audio_feat = g[key + "-" + str(i) + "/" + "audio-features"][...]
            # image_feat = f[key + "/" + "vision-features"][...]
            # pdb.set_trace()
