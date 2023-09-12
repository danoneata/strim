import os
import pdb

from pathlib import Path

import click
import h5py
import torch

from transcribe_with_images.train import (
    AudioToImageMapper,
    Flickr8kDataset,
    H5_PATH_AUDIO,
    H5_PATH_IMAGE,
    MODELS,
    get_sample,
)


@click.command()
@click.option("-i", "--image-model", "image_model_name", default="blip-base")
@click.option("-a", "--audio-model", "audio_model_name", default="wav2vec2-xls-r-2b")
@click.option("-d", "--dataset", "dataset_name", default="flickr8k")
@click.option("-m", "--mapping-model", "mapping_model_name")
def main(image_model_name, audio_model_name, dataset_name, mapping_model_name):
    device = "cuda"
    split = "test"

    dataset = Flickr8kDataset(split=split)
    audio_h5 = h5py.File(H5_PATH_AUDIO.format(audio_model_name, dataset_name, split), "r")
    image_h5 = h5py.File(H5_PATH_IMAGE.format(image_model_name, dataset_name, split), "r")

    def get_neg_loss(path: Path) -> float:
        filename_without_ext = path.stem
        *_, neg_loss = filename_without_ext.split("=")
        return float(neg_loss)

    def get_model_path():
        folder = f"output/audio-to-image-mapper/{mapping_model_name}-{dataset_name}-{audio_model_name}-{image_model_name}"
        folder = Path(folder)
        model_paths = [folder / f for f in os.listdir(folder)]
        model_path = max(model_paths, key=get_neg_loss)
        print(model_path)
        return model_path

    model = AudioToImageMapper(
        dim_audio=1920,
        dim_image=768,
        len_image_seq=577,
        **MODELS[mapping_model_name],
    )
    model = model.to(device)
    model.load_state_dict(torch.load(get_model_path()))

    for i in range(10):
        sample = get_sample(dataset, audio_h5, image_h5, i)
        audio_feat = sample["audio-feat"]
        audio_feat = audio_feat.to(device)
        audio_feat = audio_feat.unsqueeze(0)

        B, T, _ = audio_feat.shape
        assert B == 1
        padding_mask = torch.full((B, T), fill_value=False)
        padding_mask = padding_mask.to(device)

        input = audio_feat, padding_mask
        image_feat_pred = model(input)
        pdb.set_trace()

    audio_h5.close()
    image_h5.close()


if __name__ == "__main__":
    main()
