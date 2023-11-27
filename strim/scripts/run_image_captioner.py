import pdb

import click
import h5py
import torch

from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BlipProcessor,
    BlipForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration,
)


from strim.data import Flickr8kDataset


GENERATE_CONFIGS = {
    "diverse": {
        "num_return_sequences": 5,
        "num_beams": 5,
        "num_beam_groups": 5,
        "diversity_penalty": 1.0,
    },
    "sample": {
        "num_return_sequences": 5,
        "num_beams": 5,
        "do_sample": True,
    },
    "topk": {
        "num_return_sequences": 5,
        "num_beams": 5,
        "do_sample": False,
    },
}


class Blip:
    def __init__(self, name):
        self.name = name
        self.load()

    def load(self):
        self.processor = BlipProcessor.from_pretrained(self.name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.name)
        self.model = self.model.to("cuda")

    def __call__(self, raw_image, config_generate):
        inputs = self.processor(raw_image, return_tensors="pt")
        inputs = inputs.to("cuda")
        out = self.model.generate(**inputs, **config_generate)
        return self.processor.batch_decode(out, skip_special_tokens=True)


class Blip2:
    def __init__(self, name):
        self.name = name
        self.dtype = torch.float16
        self.load()

    def load(self):
        self.processor = Blip2Processor.from_pretrained(self.name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.name,
            device_map="auto",
            torch_dtype=self.dtype,
        )

    def __call__(self, raw_image, config_generate):
        inputs = self.processor(raw_image, return_tensors="pt")
        inputs = inputs.to("cuda", self.dtype)
        out = self.model.generate(**inputs, **config_generate)
        return self.processor.batch_decode(out, skip_special_tokens=True)


class Git:
    def __init__(self, name):
        self.name = name
        self.load()

    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.name)
        self.model = AutoModelForCausalLM.from_pretrained(self.name)
        self.model = self.model.to("cuda")

    def __call__(self, raw_image, config_generate):
        inputs = self.processor(images=raw_image, return_tensors="pt")
        inputs = inputs.to("cuda")
        out = self.model.generate(**inputs, **config_generate)
        return self.processor.batch_decode(out, skip_special_tokens=True)


MODEL_CONFIGS = {
    "blip-base": {
        "name": "Salesforce/blip-image-captioning-base",
        "model": Blip,
    },
    "blip-large": {
        "name": "Salesforce/blip-image-captioning-large",
        "model": Blip,
    },
    "blip2-opt-2.7b": {
        "name": "Salesforce/blip2-opt-2.7b",
        "model": Blip2,
    },
    "git-base-coco": {
        "name": "microsoft/git-base-coco",
        "model": Git,
    },
    "git-large-coco": {
        "name": "microsoft/git-large-coco",
        "model": Git,
    },
}


@click.command()
@click.option("-m", "--model", "model_name", default="blip-base")
@click.option("-g", "--generation", "generation_name", default="diverse")
@click.option("-d", "--dataset", "dataset_name", default="flickr8k")
@click.option("-s", "--split", required=True)
def main(model_name, generation_name, dataset_name, split):
    assert dataset_name == "flickr8k"
    dataset = Flickr8kDataset(split=split)
    num_samples = len(dataset)

    config = MODEL_CONFIGS[model_name]
    config_generate = GENERATE_CONFIGS[generation_name]

    model_name_full = config["name"]
    model = config["model"](model_name_full)

    path_hdf5 = f"output/image-captioner/{model_name}-{generation_name}-{dataset_name}-{split}.h5"

    with h5py.File(path_hdf5, "a") as f:
        for i in tqdm(range(num_samples)):
            sample = dataset[i]
            group_name = sample["key-image"]

            try:
                group = f.create_group(group_name)
            except ValueError:
                group = f[group_name]

            if "generated-captions" in group:
                continue

            raw_image = Image.open(sample["path-image"])
            raw_image = raw_image.convert("RGB")

            generated_captions = model(raw_image, config_generate)
            # image_features = model.vision_model(**inputs)[0]
            # image_features = image_features.detach().cpu().numpy()

            # group.create_dataset("image-features", data=image_features)
            group.create_dataset("generated-captions", data=generated_captions)

    # dry run
    # print(model.generation_config)
    # image_key_to_captions = dataset.get_image_key_to_captions()
    # for i in range(0, 100, 5):
    #     sample = dataset[i]
    #     image_key = sample["key-image"]

    #     for c in image_key_to_captions[image_key]:
    #         print(c)
    #     print("---")

    #     raw_image = Image.open(sample["path-image"])
    #     raw_image = raw_image.convert("RGB")

    #     generated_captions = model(raw_image, config_generate)

    #     for c in generated_captions:
    #         print(c)
    #     print()


if __name__ == "__main__":
    main()
