import pdb

from functools import partial
from typing import Any, Dict, List, Optional, Union

import click
import h5py  # type: ignore
import torch  # type: ignore

from torch.nn.utils.rnn import pad_sequence  # type: ignore

from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    SpeechEncoderDecoderModel,
    default_data_collator,
)
from transformers.modeling_outputs import BaseModelOutput

from transcribe_with_images.data import Flickr8kDataset
from transcribe_with_images.scripts.extract_audio_features import get_group_name

from typing import List


# class ProjectionConfig(PretrainedConfig):
#     model_type = "projection"
#
#     def __init__(
#         self,
#         input_size=1920,
#         hidden_size=4,
#         **kwargs,
#     ):
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         super().__init__(**kwargs)
#
#
# class ProjectionModel(PreTrainedModel):
#     config_class = ProjectionConfig
#
#     def __init__(self, config):
#         super().__init__(config)
#         self.model = torch.nn.Linear(config.input_size, config.hidden_size)
#
#     def forward(self, tensor, **kwargs):
#         return (self.model(tensor), )
#
#
# AutoConfig.register("projection", ProjectionConfig)
# AutoModel.register(ProjectionConfig, ProjectionModel)


H5_PATH_AUDIO = "output/audio-features/{}-{}-{}.h5"
H5_PATH_IMAGE = "output/image-captioner/{}-{}-{}.h5"


def get_audio_to_text_mapper(
    encoder_name, decoder_name, decoder_cross_attention_hidden_size
):
    model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_name,
        decoder_name,
        # decoder_cross_attention_hidden_size=decoder_cross_attention_hidden_size,
    )

    for name, param in model.encoder.named_parameters():
        param.requires_grad = False

    for name, param in model.decoder.named_parameters():
        if "crossattention" not in name:
            param.requires_grad = False
        # else:
        #     print(name, param.shape)

    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    tokenizer.pad_token = " "

    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # num_total_params = sum(p.numel() for p in model.parameters())

    # print(num_trainable_params)
    # print(num_total_params)

    return model, tokenizer


class DatasetForTrainer:
    def __init__(self, split):
        dataset_name = "flickr8k"
        audio_model_name = "wav2vec2-xls-r-2b"
        image_model_name = "blip-base"
        self.dataset = Flickr8kDataset(split=split)
        self.audio_h5 = h5py.File(
            H5_PATH_AUDIO.format(audio_model_name, dataset_name, split), "r"
        )
        self.image_h5 = h5py.File(
            H5_PATH_IMAGE.format(image_model_name, dataset_name, split), "r"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        max_audio_len = 600
        sample = self.dataset[i]

        # input: audio
        path_audio = get_group_name(sample) + "/" + "audio-features"
        audio_feat = self.audio_h5[path_audio][...]
        audio_feat = audio_feat[:max_audio_len]

        # output: text
        path_text = sample["key-image"] + "/" + "generated-caption"
        text = self.image_h5[path_text][...].item().decode()

        return {
            "encoder_outputs": torch.tensor(audio_feat),
            "labels": text,
        }


MODELS = {
    "tiny": {
        # "encoder_name": "config/projection-wav2vec2-xls-r-2b-tiny",
        # "decoder_name": "bert-base-uncased",
        "encoder_name": "facebook/wav2vec2-xls-r-2b",
        "decoder_name": "gpt2",
        "decoder_cross_attention_hidden_size": 768,
    },
    # "medium": {
    #     "encoder_name": "config/projection-wav2vec2-xls-r-2b-medium",
    #     # "decoder_name": "bert-base-uncased",
    #     "decoder_name": "gpt2",
    #     "decoder_cross_attention_hidden_size": 8,
    # },
}


# for _, m in MODELS.items():
#     config = ProjectionConfig(1920, m["decoder_cross_attention_hidden_size"])
#     config.save_pretrained(m["encoder_name"])
#
#     model = ProjectionModel(config)
#     model.save_pretrained(m["encoder_name"])

CONFIGS = {
    "00": {
        "model": MODELS["tiny"],
        # "dataset": {
        #     "name": "flickr8k",
        # },
        "training": {
            "num_train_epochs": 75,
            "per_device_train_batch_size": 32,
            "learning_rate": 4e-3,
            "gradient_accumulation_steps": 1,
            "fp16": False,
            # "save_strategy": "steps",
            # "logging_strategy": "steps",
            "logging_steps": 20,
            "save_total_limit": 10,
            "eval_steps": 100,
            "evaluation_strategy": "steps",
            "output_dir": "output/audio-to-text-mapper/00",
            "overwrite_output_dir": True,
        },
    }
}


def my_data_collator(tokenizer, data) -> Dict[str, torch.Tensor]:
    input_features = [datum["encoder_outputs"] for datum in data]
    input_features = pad_sequence(input_features, batch_first=True)

    texts = [datum["labels"] for datum in data]
    labels = tokenizer(texts, padding=True)
    labels = torch.tensor(labels["input_ids"])

    return {
        "encoder_outputs": BaseModelOutput(input_features),
        "labels": labels,
    }


@click.command()
@click.option("-c", "--config", "config_name")
def main(config_name):
    config = CONFIGS[config_name]
    model, tokenizer = get_audio_to_text_mapper(**config["model"])
    tr_dataset = DatasetForTrainer("train")
    te_dataset = DatasetForTrainer("dev")

    training_args = Seq2SeqTrainingArguments(**config["training"])
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tr_dataset,
        eval_dataset=te_dataset,
        data_collator=lambda features: my_data_collator(tokenizer, features),
        # tokenizer=feature_extractor,
    )

    trainer.train()


if __name__ == "__main__":
    main()
