import pdb

import click
import torch

from itertools import groupby

from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    AutoProcessor,
    SpeechEncoderDecoderModel,
)
from transformers.modeling_outputs import BaseModelOutput

from transcribe_with_images.audio_to_text.cross_attention.train import DatasetForTrainer
from transcribe_with_images.audio_to_text.prompt_tunning.train import (
    CONFIGS,
    PromptTuningModel,
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
    model_path = "output/audio-to-text-mapper-prompt-tunning/00/checkpoint-107500/pytorch_model.bin"
    # model_path = "output/audio-to-text-mapper/01/checkpoint-21500/pytorch_model.bin"

    config = CONFIGS[config_name]
    model = PromptTuningModel(**config["model"])
    tokenizer = model.text_processor

    dataset = DatasetForTrainer(split)

    CAPTION_LENGTH = 32
    generation_kwargs = {
        "max_new_tokens": CAPTION_LENGTH,
        "no_repeat_ngram_size": 2,
        # "length_penalty": 0.0,
        "num_beams": 5,
        "early_stopping": True,
        "eos_token_id": tokenizer.eos_token_id,
    }

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    samples = [dataset.dataset[i] for i in range(5 * 100)]
    samples = sorted(samples, key=lambda x: x["key-image"])
    image_groups = groupby(samples, key=lambda x: x["key-image"])
    image_groups = [(k, list(g)) for k, g in image_groups]

    import random

    random.shuffle(image_groups)

    for image_id, image_group in image_groups[:16]:
        image_group = list(image_group)
        sample = image_group[0]
        generated_captions = dataset.load_generated_captions(sample)

        print("image id:", image_id)
        print("image captions:")
        for i, caption in enumerate(generated_captions, start=1):
            print("{:2d} · {}".format(i, caption))
        print("---")

        for sample in image_group:
            audio_feats = dataset.load_audio_features(sample)

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
                    encoder_outputs=encoder_outputs, **generation_kwargs
                )
                out = tokenizer.batch_decode(out, skip_special_tokens=True)

            audio_transcript = sample["text"]
            audio_transcript = audio_transcript.lower().replace(" .", "")
            print("{} · {}".format("input (audio)", audio_transcript))
            print("{} · {}".format("ouput (text) ", out[0]))
            print("---")
            pdb.set_trace()

        print()
        # pdb.set_trace()


if __name__ == "__main__":
    main()
