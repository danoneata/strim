import os

from itertools import groupby
from pathlib import Path
from typing import Union, Literal

from transcribe_with_images.utils import read_file


BASE_DATA_DIR = Path("data")
Split = Literal["train", "dev", "test"]


class Flickr8kDataset:
    def __init__(self, *, split: Split):
        self.split = split
        self.audio_dir = BASE_DATA_DIR / "flickr8k" / "audio" / "wavs"
        self.image_dir = BASE_DATA_DIR / "flickr8k" / "image" / "Flicker8k_Dataset"
        self.text_dir = BASE_DATA_DIR / "flickr8k" / "text"
        self.data = self.load_data(split)

    def load_data(self, split):
        def parse_token(line):
            key, *words = line.strip().split()
            text = " ".join(words)
            img, i = key.split("#")
            key_image = img.split(".")[0]
            key_audio = (key_image, int(i))
            return {
                "key-image": key_image,
                "key-audio": key_audio,
                "text": text,
            }

        def get_image_keys(split):
            path_img = str(self.text_dir / f"Flickr_8k.{split}Images.txt")
            return set(read_file(path_img, lambda line: line.split(".")[0]))

        selected_keys = get_image_keys(split)
        file_transcript = str(self.text_dir / "Flickr8k.token.txt")
        data = read_file(file_transcript, parse_token)
        data = [sample for sample in data if sample["key-image"] in selected_keys]
        return data

    def get_image_key_to_captions(self):
        func = lambda sample: sample["key-image"]
        data = sorted(self.data, key=func)
        groups = groupby(data, key=func)
        return {
            k: [sample["text"] for sample in group]
            for k, group in groups
        }

    def get_audio_path(self, sample):
        key, i = sample["key-audio"]
        return self.audio_dir / (key + "_" + str(i) + ".wav")

    def get_image_path(self, sample):
        image_name = sample["key-image"]
        return self.image_dir / (image_name + ".jpg")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sample = self.data[i]
        return {
            **sample,
            "path-audio": str(self.get_audio_path(sample)),
            "path-image": str(self.get_image_path(sample)),
        }
