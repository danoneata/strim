import click
import pdb

import h5py
import librosa
import numpy as np
import soundfile as sf
import torch

from transformers import AutoFeatureExtractor, WavLMModel, Wav2Vec2Model
from tqdm import tqdm

from strim.data import DATASETS


SAMPLING_RATE = 16_000


class HuggingFaceFeatureExtractor:
    def __init__(self, model_class, name):
        self.device = "cuda"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(name)
        self.model = model_class.from_pretrained(name)
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, audio, sr):
        max_length = 10 * sr
        inputs = self.feature_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            # padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                # output_attentions=True,
                # output_hidden_states=False,
            )
        return outputs.last_hidden_state


FEATURE_EXTRACTORS = {
    "wav2vec2-base": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-base"
    ),
    "wav2vec2-large": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-large"
    ),
    "wav2vec2-large-lv60": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-large-lv60"
    ),
    "wav2vec2-large-robust": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-large-robust"
    ),
    "wav2vec2-large-xlsr-53": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-large-xlsr-53"
    ),
    "wav2vec2-xls-r-300m": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-xls-r-300m"
    ),
    "wav2vec2-xls-r-1b": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-xls-r-1b"
    ),
    "wav2vec2-xls-r-2b": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-xls-r-2b"
    ),
    "wavlm-base": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "microsoft/wavlm-base"
    ),
    "wavlm-base-sv": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "microsoft/wavlm-base-sv"
    ),
    "wavlm-base-plus": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "microsoft/wavlm-base-plus"
    ),
    "wavlm-large": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "microsoft/wavlm-large"
    ),
}


def get_group_name(sample):
    key, i = sample["key-audio"]
    return f"{key}-{i}"


@click.command()
@click.option("-m", "--model", "model_name", default="wav2vec2-xls-r-2b")
@click.option("-d", "--dataset", "dataset_name", default="flickr8k")
@click.option("-s", "--split", required=True)
def main(
    model_name: str,
    dataset_name: str,
    split: str,
):
    dataset = DATASETS[dataset_name](split=split)
    num_samples = len(dataset)

    feature_extractor = FEATURE_EXTRACTORS[model_name]()

    def extract1(audio):
        feature = feature_extractor(audio, sr=SAMPLING_RATE)
        feature = feature[0].cpu().numpy()
        return feature

    path_hdf5 = f"output/audio-features/{model_name}-{dataset_name}-{split}.h5"

    with h5py.File(path_hdf5, "a") as f:
        for i in tqdm(range(num_samples)):
            sample = dataset[i]
            group_name = get_group_name(sample)

            try:
                group = f.create_group(group_name)
            except ValueError:
                group = f[group_name]

            if "audio-features" in group:
                continue

            # audio1, sr1 = sf.read(sample["path-audio"])

            audio, sr = librosa.load(sample["path-audio"], mono=True, sr=SAMPLING_RATE)
            audio = audio.astype(np.float64)
            audio = audio.T

            # pdb.set_trace()

            assert sr == SAMPLING_RATE

            audio = torch.from_numpy(audio).to(feature_extractor.device)
            audio_features = extract1(audio)
            group.create_dataset("audio-features", data=audio_features)


if __name__ == "__main__":
    main()
