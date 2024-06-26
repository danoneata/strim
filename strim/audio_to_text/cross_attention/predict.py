import json
import pdb

from pathlib import Path

import click
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
    "00-blip2-opt-2.7b-diverse-best": {
        "model-path": "output/audio-to-text-mapper/00-blip2-opt-2.7b-diverse/checkpoint-16000/pytorch_model.bin",
        "dataset-name": "flickr8k",
    },
    "00-yfacc-best": {
        "model-path": "output/audio-to-text-mapper/00-yfacc/checkpoint-1500/pytorch_model.bin",
        "dataset-name": "yfacc",
    },
    "00-transcripts-best": {
        "model-path": "output/audio-to-text-mapper/00-transcripts/checkpoint-16000/pytorch_model.bin",
        "dataset-name": "flickr8k",
    },
    "00-yfacc-transcripts-best": {
        "model-path": "output/audio-to-text-mapper/00-yfacc-transcripts/checkpoint-500/pytorch_model.bin",
        "dataset-name": "yfacc",
    },
}


def get_config_predict(name, tr_config_name):
    try:
        return CONFIGS_PREDICT[name]
    except KeyError:
        def get_model_path():
            folder = Path(f"output/audio-to-text-mapper") / tr_config_name
            subfolder, *rest = [f for f in folder.iterdir() if f.is_dir()]
            assert not rest
            return subfolder / "pytorch_model.bin"
        config = CONFIGS[tr_config_name]
        return {
            "model-path": get_model_path(),
            "dataset-name": config["dataset"]["name"],
        }


@click.command()
@click.option("-c", "--config", "config_name")
@click.option("-p", "--config-predict", "config_predict_name", required=False)
def main(config_name, config_predict_name=None):
    audio_model_name = "wav2vec2-xls-r-2b"
    split = "test"

    config = CONFIGS[config_name]
    config_predict = get_config_predict(config_predict_name, config_name)

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
    }

    predictions = []

    for i in tqdm(range(num_samples)):
        sample = dataset[i]
        audio_feats = load_audio_feats(sample)

        encoder_outputs = audio_feats.unsqueeze(0).to(DEVICE)
        encoder_outputs = BaseModelOutput(encoder_outputs)

        # input_ids = tokenizer([datum["labels"].lower()], return_tensors="pt")
        # input_ids = input_ids.input_ids.to(DEVICE)

        with torch.no_grad():
            # out1 = model.forward(encoder_outputs=encoder_outputs, decoder_input_ids=input_ids)
            # idxs = torch.argmax(out1.logits, dim=-1)
            # out1 = tokenizer.batch_decode(idxs, skip_special_tokens=True)
            # print(datum["labels"])
            # print(out1)

            out = model.generate(
                encoder_outputs=encoder_outputs,
                **generation_kwargs,
            )
            out = tokenizer.batch_decode(out, skip_special_tokens=True)

        prediction = {
            "key-audio": sample["key-audio"],
            "audio-transcript": sample["text"],
            "text-prediction": out[0],
        }
        predictions.append(prediction)

    path_predictions = f"output/audio-to-text-mapper/predictions/{config_name}-{config_predict_name}.json"
    with open(path_predictions, "w") as f:
        json.dump(predictions, f, indent=4)


if __name__ == "__main__":
    main()
