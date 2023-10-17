import pdb
import os

from pathlib import Path
from typing import Callable, List, Tuple, Union

import click
import torch  # type: ignore

import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    SpeechEncoderDecoderModel,
)

from transcribe_with_images.audio_to_text.train import (
    DatasetForTrainer,
    my_data_collator,
)

from typing import List


def accumulate_padding(
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    padding_side: str = "right",
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert padding_side in ["right", "left"]

    new_input_embeds = torch.empty_like(input_embeds)
    new_attention_masks = torch.empty_like(attention_mask)

    for i, (embed, mask) in enumerate(zip(input_embeds, attention_mask)):
        padding_indices = torch.where(mask == 0)[0]
        non_padding_indices = torch.where(mask == 1)[0]
        if padding_side == "left":
            new_indices = torch.cat((padding_indices, non_padding_indices), dim=0)
        else:
            new_indices = torch.cat((non_padding_indices, padding_indices), dim=0)
        new_input_embeds[i] = embed.index_select(0, new_indices)
        new_attention_masks[i] = mask.index_select(0, new_indices)

    return new_input_embeds, new_attention_masks


class LanguageDecoder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        self.forward = self.model.forward
        self.generate = self.model.generate

    @property
    def model_id(self) -> str:
        return type(self.model).__name__.lower()

    @property
    def embed_dim(self) -> int:
        if "gpt" in self.model_id:
            return self.model.config.n_embd
        elif "opt" in self.model_id:
            return self.model.config.word_embed_proj_dim
        else:
            raise NotImplementedError

    @property
    def embed_tokens(self) -> nn.Module:
        if "gpt" in self.model_id:
            return self.model.transformer.wte
        elif "opt" in self.model_id:
            return self.model.model.decoder.embed_tokens
        else:
            raise NotImplementedError

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask,
        audio_embeds,
        past_key_values=None,
        use_cache=None,
        **kwargs,
    ):
        pdb.set_trace()
        expand_size = input_ids.size(0) // audio_embeds.size(0)
        audio_embeds = audio_embeds.repeat_interleave(expand_size, dim=0)
        visual_mask = torch.ones(
            audio_embeds.shape[:2], dtype=torch.long, device=audio_embeds.device
        )

        if input_ids[0][0] == self.model.config.bos_token_id:
            input_ids = input_ids[:, 1:]
            attention_mask = attention_mask[:, 1:]

        token_embeds = self.embed_tokens(input_ids)

        input_embeds = torch.cat([audio_embeds, token_embeds], dim=1)
        attention_mask = torch.cat([visual_mask, attention_mask], dim=1)

        input_embeds, attention_mask = accumulate_padding(
            input_embeds, attention_mask, padding_side="left"
        )

        if past_key_values:
            input_embeds = input_embeds[:, -1].unsqueeze(1)

        return {
            "inputs_embeds": input_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }


class MappingNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        output_length: int = 32,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        proj_bias: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.down = nn.Linear(input_dim, hidden_dim, bias=proj_bias)
        self.up = nn.Linear(hidden_dim, output_dim, bias=proj_bias)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.const = nn.Parameter(torch.randn(output_length, hidden_dim))
        # self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        L = self.const.shape[0]
        mask = torch.ones((B, L), device=x.device)
        mask = torch.cat((attention_mask, mask), dim=1)
        mask = 1 - mask

        x = self.down(x)
        x = torch.cat((x, self.const.unsqueeze(0).expand(x.size(0), -1, -1)), dim=1)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x[:, -self.const.size(0) :]
        x = self.up(x)
        # x = self.norm(x)
        return x


class PromptTuningModel(torch.nn.Module):
    def __init__(self, decoder_name: str, audio_dim: int) -> None:
        super().__init__()

        self.text_processor = AutoTokenizer.from_pretrained(decoder_name)

        if self.text_processor._pad_token is None:
            self.text_processor.pad_token = self.text_processor.eos_token

        self.language_decoder = LanguageDecoder(
            AutoModelForCausalLM.from_pretrained(
                decoder_name,
                # torch_dtype=torch.float16,
                # revision="float16",
                # low_cpu_mem_usage=True,
            )
        )

        for param in self.language_decoder.parameters():
            param.requires_grad = False

        self.mapper = MappingNetwork(
            input_dim=audio_dim,
            output_dim=self.language_decoder.embed_dim,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Union[str, Path],
        dtype: Union[str, torch.dtype] = None,
        **kwargs,
    ) -> nn.Module:
        with torch_dtype(dtype):
            model = cls(**kwargs)

        logger.info(f"Loading mapper weights from {checkpoint_path}")
        if is_remote_url(checkpoint_path):
            state_dict = torch.hub.load_state_dict_from_url(
                checkpoint_path, map_location="cpu"
            )
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.mapper.load_state_dict(state_dict)

        return model

    def embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            token_embeds = self.language_decoder.embed_tokens(input_ids)
        return token_embeds

    def forward(
        self,
        # audio_embeds: torch.Tensor,
        # target_ids: torch.Tensor,
        # prefix_ids: torch.Tensor = None,
        encoder_outputs: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        audio_embeds = self.mapper(encoder_outputs.last_hidden_state, attention_mask)
        input_embeds = self.embed_text(input_ids)
        input_embeds_full = torch.cat((audio_embeds, input_embeds), dim=1)

        audio_mask = torch.ones(
            audio_embeds.shape[:2],
            dtype=torch.long,
            device=audio_embeds.device,
        )
        decoder_attention_mask_full = torch.cat((audio_mask, decoder_attention_mask), dim=1)

        labels = input_ids.clone().detach()
        labels[labels == self.text_processor.pad_token_id] = -100
        labels_full = torch.cat((-100 * audio_mask, labels), dim=1)

        outputs = self.language_decoder(
            inputs_embeds=input_embeds_full,
            attention_mask=decoder_attention_mask_full,
            labels=labels_full,
        )

        return outputs

    @torch.inference_mode()
    def generate(
        self, pixel_values: torch.Tensor, input_ids: torch.Tensor = None, **kwargs
    ) -> List[str]:
        audio_embeds = self.embed_image(pixel_values)
        if input_ids is None:
            input_ids = torch.full(
                (audio_embeds.size(0), 1),
                self.text_processor.bos_token_id,
                dtype=torch.long,
                device=audio_embeds.device,
            )
        attention_mask = (input_ids != self.text_processor.pad_token_id).long()

        output_ids = self.language_decoder.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            audio_embeds=audio_embeds,
            eos_token_id=self.text_processor.get_vocab()["."],
            pad_token_id=self.text_processor.pad_token_id,
            **kwargs,
        )
        output_ids = output_ids[:, input_ids.size(1) :]

        return output_ids

    def image_transform(self, image: Image.Image, **kwargs) -> torch.Tensor:
        return self.image_processor(
            image, return_tensors="pt", **kwargs
        ).pixel_values.squeeze(0)

    def text_transform(self, text: Union[str, List[str]], **kwargs) -> torch.Tensor:
        return self.text_processor(
            text, padding="longest", return_tensors="pt", **kwargs
        )


MODELS = {
    "tiny": {
        "decoder_name": "gpt2",
        "audio_dim": 1920,
    },
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
            "remove_unused_columns": False,
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


@click.command()
@click.option("-c", "--config", "config_name")
def main(config_name):
    config = CONFIGS[config_name]
    model = PromptTuningModel(**config["model"])
    tokenizer = model.text_processor
    tr_dataset = DatasetForTrainer("train")
    te_dataset = DatasetForTrainer("dev")

    output_dir = os.path.join("output/audio-to-text-mapper-prompt-tunning", config_name)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir, **config["training"]
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
