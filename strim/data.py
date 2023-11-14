import os

from itertools import groupby
from pathlib import Path
from typing import Union, Literal
from toolz import join

from strim.utils import read_file


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


class YFACCDataset:
    def __init__(self, *, split: Split):
        self.split = split
        self.audio_dir = BASE_DATA_DIR / "yfacc" / "audio"
        self.image_dir = BASE_DATA_DIR / "flickr8k" / "image" / "Flicker8k_Dataset"
        self.text_en_dir = BASE_DATA_DIR / "flickr8k" / "text"
        self.text_yo_dir = BASE_DATA_DIR / "yfacc" / "text"
        self.data = self.load_data(split)

    @staticmethod
    def _parse_audio_path(path: Path):
        name = path.stem
        prefix, *keys, i = name.split("_")
        assert prefix == "S001"
        key = "_".join(keys)
        return key, int(i)

    def load_data(self, split):
        def parse_token(line):
            key, *words = line.strip().split()
            text = " ".join(words)
            img, i = key.split("#")
            key_image = img.split(".")[0]
            try:
                key_audio = (key_image, int(i))
            except ValueError:
                print("WARN: Incorrect audio key")
                print("LINE:", line)
                key_audio = (key_image, 0)
            return {
                "key-image": key_image,
                "key-audio": key_audio,
                "text": text,
            }

        def get_audio_keys(split):
            files = (self.audio_dir / split).iterdir()
            return set(self._parse_audio_path(path) for path in files)

        selected_keys = get_audio_keys(split)

        file_transcript_en = str(self.text_en_dir / "Flickr8k.token.txt")
        file_transcript_yo = str(self.text_yo_dir / f"Flickr8k.token.{split}_yoruba.txt")

        data_en = read_file(file_transcript_en, parse_token)
        data_yo = read_file(file_transcript_yo, parse_token)
        data = [
            {
                **datum_en,
                "text-yo": datum_yo["text"],
            }
            for datum_en, datum_yo in join("key-audio", data_en, "key-audio", data_yo)
        ]

        data = [sample for sample in data if sample["key-audio"] in selected_keys]

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
        return self.audio_dir / self.split / ("S001" + "_" + key + "_" + str(i) + ".wav")

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


DATASETS = {
    "flickr8k": Flickr8kDataset,
    "yfacc": YFACCDataset,
}