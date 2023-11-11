import pdb
import os

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

from strim.data import Flickr8kDataset
from strim.scripts.extract_audio_features import get_group_name

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
        self.num_generated_captions_per_image = 5
        self.dataset = Flickr8kDataset(split=split)
        self.audio_h5 = h5py.File(
            H5_PATH_AUDIO.format(audio_model_name, dataset_name, split), "r"
        )
        self.image_h5 = h5py.File(
            H5_PATH_IMAGE.format(image_model_name, dataset_name, split), "r"
        )

    def __len__(self):
        return self.num_generated_captions_per_image * len(self.dataset)

    def load_audio_features(self, sample):
        max_audio_len = 600
        path_audio = get_group_name(sample) + "/" + "audio-features"
        audio_feat = self.audio_h5[path_audio][...]
        audio_feat = torch.tensor(audio_feat[:max_audio_len])
        return audio_feat

    def load_generated_captions(self, sample):
        path_text = sample["key-image"] + "/" + "generated-captions"
        texts = self.image_h5[path_text][...]
        texts = [text.decode() for text in texts]
        return texts

    def __getitem__(self, i):
        sample_idx = i // self.num_generated_captions_per_image
        caption_idx = i % self.num_generated_captions_per_image

        sample = self.dataset[sample_idx]

        audio_feats = self.load_audio_features(sample)
        target_text = self.load_generated_captions(sample)[caption_idx]

        # text = sample["text"].lower()
        # words = text.split()
        # words = [w for w in words if w != "."]
        # text = " ".join(words)

        return {
            "encoder_outputs": audio_feats,
            "labels": target_text,
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
            "num_train_epochs": 50,
            "per_device_train_batch_size": 20,
            "learning_rate": 1e-4,
            "gradient_accumulation_steps": 1,
            "fp16": False,
            # "save_strategy": "steps",
            # "logging_strategy": "steps",
            "warmup_steps": 400,
            "logging_steps": 20,
            "save_total_limit": 1,
            "eval_steps": 100,
            "save_steps": 500,
            "evaluation_strategy": "steps",
            "overwrite_output_dir": True,
            "predict_with_generate": True,
            "generation_num_beams": 1,
            "load_best_model_at_end": True,
        },
    },
    "01": {
        "model": MODELS["tiny"],
        # "dataset": {
        #     "name": "flickr8k",
        # },
        "training": {
            "num_train_epochs": 150,
            "per_device_train_batch_size": 16,
            "learning_rate": 1e-4,
            "gradient_accumulation_steps": 1,
            "fp16": False,
            # "save_strategy": "steps",
            # "logging_strategy": "steps",
            "warmup_steps": 400,
            "logging_steps": 20,
            "save_total_limit": 1,
            "eval_steps": 100,
            "save_steps": 500,
            "evaluation_strategy": "steps",
            "overwrite_output_dir": True,
            "predict_with_generate": True,
            "generation_num_beams": 1,
            "load_best_model_at_end": True,
        },
    },
}


def my_data_collator(tokenizer, data) -> Dict[str, torch.Tensor]:
    input_features = [datum["encoder_outputs"] for datum in data]
    input_features = pad_sequence(input_features, batch_first=True)

    padding_mask = [
        torch.full((datum["encoder_outputs"].shape[0],), fill_value=1) for datum in data
    ]
    padding_mask = pad_sequence(padding_mask, batch_first=True, padding_value=0)

    texts = [datum["labels"] for datum in data]
    texts_padded = tokenizer(texts, padding=True)

    input_ids = torch.tensor(texts_padded["input_ids"])
    decoder_attention_mask = torch.tensor(texts_padded["attention_mask"])

    labels = input_ids.clone().detach()
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "encoder_outputs": BaseModelOutput(input_features),
        "attention_mask": padding_mask,
        "labels": labels,
        "decoder_input_ids": input_ids,
        "decoder_attention_mask": decoder_attention_mask,
    }


@click.command()
@click.option("-c", "--config", "config_name")
def main(config_name):
    config = CONFIGS[config_name]
    model, tokenizer = get_audio_to_text_mapper(**config["model"])
    tr_dataset = DatasetForTrainer("train")
    te_dataset = DatasetForTrainer("dev")

    output_dir = os.path.join("output/audio-to-text-mapper", config_name)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        **config["training"],
    )
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
