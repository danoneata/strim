import pdb

import click
import torch

from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    AutoProcessor,
    SpeechEncoderDecoderModel,
)
from transformers.modeling_outputs import BaseModelOutput

from transcribe_with_images.audio_to_text.train import CONFIGS, DatasetForTrainer, get_audio_to_text_mapper


DEVICE = "cuda"


# def load_model(checkpoint_path):
#     config = AutoConfig.from_pretrained(checkpoint_path + "/config.json")
#     model = SpeechEncoderDecoderModel.from_pretrained(checkpoint_path)
#     model.config = config
#     return model


@click.command()
@click.option("-c", "--config", "config_name")
def main(config_name):
    split = "train"
    model_path = "output/audio-to-text-mapper/00/checkpoint-1500/pytorch_model.bin"

    config = CONFIGS[config_name]
    model, tokenizer = get_audio_to_text_mapper(**config["model"])
    dataset = DatasetForTrainer(split)

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    for i in range(64):
        datum = dataset[i]
        encoder_outputs = datum["encoder_outputs"].unsqueeze(0).to(DEVICE)
        encoder_outputs = BaseModelOutput(encoder_outputs)
        with torch.no_grad():
            out = model.generate(encoder_outputs=encoder_outputs)
            out = tokenizer.batch_decode(out, skip_special_tokens=True)
        print(dataset.dataset[i]["text"])
        print(datum["labels"])
        print(out)
        print()


if __name__ == "__main__":
    main()