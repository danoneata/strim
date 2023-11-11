import os
import pdb

from pathlib import Path

import click
import h5py
import torch

from transformers import BlipProcessor, BlipForConditionalGeneration

from strim.audio_to_image.train import (
    AudioToImageMapper,
    Flickr8kDataset,
    H5_PATH_AUDIO,
    H5_PATH_IMAGE,
    MODELS,
    get_sample,
)


def generate(self, image_embeds, **generate_kwargs):
    batch_size = image_embeds.size(0)
    image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        image_embeds.device
    )

    input_ids = (
        torch.LongTensor(
            [[self.decoder_input_ids, self.config.text_config.eos_token_id]]
        )
        .repeat(batch_size, 1)
        .to(image_embeds.device)
    )

    input_ids[:, 0] = self.config.text_config.bos_token_id
    attention_mask = None

    outputs = self.text_decoder.generate(
        input_ids=input_ids[:, :-1],
        eos_token_id=self.config.text_config.sep_token_id,
        pad_token_id=self.config.text_config.pad_token_id,
        attention_mask=attention_mask,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_attention_mask,
        **generate_kwargs,
    )

    return outputs


def generate_caption(model, preprocessor, image_embeds, **generate_kwargs):
    out = generate(model, image_embeds, **generate_kwargs)
    return preprocessor.decode(out[0], skip_special_tokens=True)


def get_neg_loss(path: Path) -> float:
    filename_without_ext = path.stem
    *_, neg_loss = filename_without_ext.split("=")
    return float(neg_loss)

def get_epoch(path: Path) -> float:
    filename_without_ext = path.stem
    *_, epoch, _ = filename_without_ext.split("_")
    return int(epoch)

def get_model_path(image_model_name, audio_model_name, dataset_name, mapping_model_name):
    folder = f"output/audio-to-image-mapper/{mapping_model_name}-{dataset_name}-{audio_model_name}-{image_model_name}"
    folder = Path(folder)
    model_paths = [folder / f for f in os.listdir(folder)]
    model_path = max(model_paths, key=get_neg_loss)
    print(model_path)
    return model_path


@click.command()
@click.option("-i", "--image-model", "image_model_name", default="blip-base")
@click.option("-a", "--audio-model", "audio_model_name", default="wav2vec2-xls-r-2b")
@click.option("-d", "--dataset", "dataset_name", default="flickr8k")
@click.option("-m", "--mapping-model", "mapping_model_name")
def main(image_model_name, audio_model_name, dataset_name, mapping_model_name):
    device = "cuda"
    split = "train"

    dataset = Flickr8kDataset(split=split)
    audio_h5 = h5py.File(
        H5_PATH_AUDIO.format(audio_model_name, dataset_name, split), "r"
    )
    image_h5 = h5py.File(
        H5_PATH_IMAGE.format(image_model_name, dataset_name, split), "r"
    )

    model_path = get_model_path(image_model_name, audio_model_name, dataset_name, mapping_model_name)
    model = AudioToImageMapper(
        dim_audio=1920,
        dim_image=768,
        len_image_seq=577,
        **MODELS[mapping_model_name],
    )
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    assert image_model_name == "blip-base"
    image_model_name_full = "Salesforce/blip-image-captioning-base"
    image_captioning_processor = BlipProcessor.from_pretrained(image_model_name_full)
    image_captioning_model = BlipForConditionalGeneration.from_pretrained(
        image_model_name_full
    )
    image_captioning_model = image_captioning_model.to(device)

    for i in range(32):
        sample = get_sample(dataset, audio_h5, image_h5, i)

        image_feat = sample["image-feat"]
        image_feat = image_feat.to(device)
        # image_feat = image_feat.unsqueeze(0)

        audio_feat = sample["audio-feat"]
        audio_feat = audio_feat.to(device)
        audio_feat = audio_feat.unsqueeze(0)

        B, T, _ = audio_feat.shape
        assert B == 1
        padding_mask = torch.full((B, T), fill_value=False)
        padding_mask = padding_mask.to(device)

        input = audio_feat, padding_mask
        image_feat_pred = model(input)

        generated_caption_1 = generate_caption(
            image_captioning_model,
            image_captioning_processor,
            image_feat_pred,
        )
        generated_caption_2 = generate_caption(
            image_captioning_model,
            image_captioning_processor,
            image_feat,
        )
        print(generated_caption_1)
        print(generated_caption_2)
        print()

    audio_h5.close()
    image_h5.close()


if __name__ == "__main__":
    main()
