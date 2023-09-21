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

from transcribe_with_images.audio_to_text.train import (
    CONFIGS,
    DatasetForTrainer,
    get_audio_to_text_mapper,
)


DEVICE = "cuda"


# def load_model(checkpoint_path):
#     config = AutoConfig.from_pretrained(checkpoint_path + "/config.json")
#     model = SpeechEncoderDecoderModel.from_pretrained(checkpoint_path)
#     model.config = config
#     return model


@click.command()
@click.option("-c", "--config", "config_name")
def main(config_name):
    split = "test"
    model_path = "output/audio-to-text-mapper/00/checkpoint-3500/pytorch_model.bin"
    # model_path = "output/audio-to-text-mapper/01/checkpoint-21500/pytorch_model.bin"

    config = CONFIGS[config_name]
    model, tokenizer = get_audio_to_text_mapper(**config["model"])
    dataset = DatasetForTrainer(split)

    CAPTION_LENGTH = 32
    generation_kwargs = {
        "max_new_tokens": CAPTION_LENGTH,
        "no_repeat_ngram_size": 0,
        "length_penalty": 0.0,
        "num_beams": 5,
        "early_stopping": True,
        "eos_token_id": tokenizer.eos_token_id,
    }

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    for i in range(0, 30, 5):
        datum = dataset[i]

        encoder_outputs = datum["encoder_outputs"].unsqueeze(0).to(DEVICE)
        encoder_outputs = BaseModelOutput(encoder_outputs)

        # input_ids = tokenizer([datum["labels"].lower()], return_tensors="pt")
        # input_ids = input_ids.input_ids.to(DEVICE)

        with torch.no_grad():
            # out1 = model.forward(encoder_outputs=encoder_outputs, decoder_input_ids=input_ids)
            # idxs = torch.argmax(out1.logits, dim=-1)
            # out1 = tokenizer.batch_decode(idxs, skip_special_tokens=True)
            # print(datum["labels"])
            # print(out1)

            out = model.generate(encoder_outputs=encoder_outputs, **generation_kwargs)
            out = tokenizer.batch_decode(out, skip_special_tokens=True)

        audio_transcript = dataset.dataset[i]["text"]
        audio_transcript = audio_transcript.lower().replace(" .", "")
        print("{:9s} · {}".format("audio", audio_transcript))
        print("{:9s} · {}".format("caption", datum["labels"]))
        print("{:9s} · {}".format("generated", out[0]))
        print()
        # pdb.set_trace()


if __name__ == "__main__":
    main()
