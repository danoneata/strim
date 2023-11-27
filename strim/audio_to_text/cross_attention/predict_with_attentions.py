import json
import os
import pdb

from pathlib import Path

import click
import numpy as np
import torch

from tqdm import tqdm

from transformers.modeling_outputs import BaseModelOutput

from strim.data import DATASETS
from strim.audio_to_text.cross_attention.train import (
    CONFIGS,
    AudioFeaturesLoader,
    get_audio_to_text_mapper,
)


DEVICE = "cuda"


# def load_model(checkpoint_path):
#     config = AutoConfig.from_pretrained(checkpoint_path + "/config.json")
#     model = SpeechEncoderDecoderModel.from_pretrained(checkpoint_path)
#     model.config = config
#     return model


CONFIGS_PREDICT = {
    "00-best-2023-11-11": {
        "model-path": "output/audio-to-text-mapper/00/checkpoint-16500/pytorch_model.bin",
        "dataset-name": "flickr8k",
    },
    "00-best-2023-09-21": {
        "model-path": "output/audio-to-text-mapper/2023-09-21/00/checkpoint-15000/pytorch_model.bin",
        "dataset-name": "flickr8k",
    },
    "00-yfacc-best": {
        "model-path": "output/audio-to-text-mapper/00-yfacc/checkpoint-1500/pytorch_model.bin",
        "dataset-name": "yfacc",
    },
}


@click.command()
@click.option("-c", "--config", "config_name")
@click.option("-p", "--config-predict", "config_predict_name")
def main(config_name, config_predict_name):
    audio_model_name = "wav2vec2-xls-r-2b"
    split = "test"

    config = CONFIGS[config_name]
    config_predict = CONFIGS_PREDICT[config_predict_name]

    dataset_name = config_predict["dataset-name"]
    dataset = DATASETS[dataset_name](split=split)
    num_samples = len(dataset)
    load_audio_feats = AudioFeaturesLoader(audio_model_name, dataset_name, split)

    model_path = config_predict["model-path"]
    model, tokenizer = get_audio_to_text_mapper(**config["model"])
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    CAPTION_LENGTH = 32
    generation_kwargs = {
        "max_new_tokens": CAPTION_LENGTH,
        "no_repeat_ngram_size": 2,
        # "length_penalty": 0.0,
        "num_beams": 5,
        "early_stopping": True,
        "eos_token_id": tokenizer.eos_token_id,
        # "output_attentions": True,
        # "return_dict_in_generate": True,
    }

    predictions = []

    output_dir = Path(f"output/audio-to-text-mapper/predictions-with-attentions")
    cross_attentions_dir = output_dir / f"{config_name}-{config_predict_name}-cross-attentions"
    cross_attentions_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(num_samples)):
        sample = dataset[i]
        audio_feats = load_audio_feats(sample)

        encoder_outputs = audio_feats.unsqueeze(0).to(DEVICE)
        encoder_outputs = BaseModelOutput(encoder_outputs)

        with torch.no_grad():
            out = model.generate(
                encoder_outputs=encoder_outputs,
                **generation_kwargs,
            )
            out = out[0]
            out_text = tokenizer.decode(out, skip_special_tokens=True)
            out_tokens = tokenizer.convert_ids_to_tokens(out)
            outputs = model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=out,
                return_dict=True,
                output_attentions=True,
            )
            cross_attention = outputs.cross_attentions
            cross_attention = [ca[0].cpu().numpy() for ca in cross_attention]
            cross_attention = np.array(cross_attention)

            np.save(
                cross_attentions_dir / f"{i}.npy",
                cross_attention,
            )

        prediction = {
            "key-audio": sample["key-audio"],
            "audio-transcript": sample["text"],
            "text-prediction": out_text,
            "text-tokens": out_tokens,
        }
        predictions.append(prediction)

    path_predictions = f"output/audio-to-text-mapper/predictions-with-attentions/{config_name}-{config_predict_name}.json"
    with open(path_predictions, "w") as f:
        json.dump(predictions, f, indent=4)


if __name__ == "__main__":
    main()
