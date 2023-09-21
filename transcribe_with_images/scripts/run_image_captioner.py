import pdb

import click
import h5py

from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

from transcribe_with_images.data import Flickr8kDataset


MODEL_CONFIGS = {
    "blip-base": {
        "full-name": "Salesforce/blip-image-captioning-base",
        "generate-kwargs": {
            "num_return_sequences": 5,
            "num_beams": 5,
            "num_beam_groups": 5,
            "diversity_penalty": 1.0,
        },
    },
    "blip-large": {
        "full-name": "Salesforce/blip-image-captioning-large",
        "generate-kwargs": {
            "num_return_sequences": 3,
            "num_beams": 5,
            "num_beam_groups": 5,
            "diversity_penalty": 1.0,
        },
    },
}


@click.command()
@click.option("-m", "--model", "model_name", default="blip-base")
@click.option("-d", "--dataset", "dataset_name", default="flickr8k")
@click.option("-s", "--split", required=True)
def main(model_name, dataset_name, split):
    assert dataset_name == "flickr8k"
    dataset = Flickr8kDataset(split=split)
    num_samples = len(dataset)

    config = MODEL_CONFIGS[model_name]
    model_name_full = config["full-name"]
    processor = BlipProcessor.from_pretrained(model_name_full)
    model = BlipForConditionalGeneration.from_pretrained(model_name_full)
    model = model.to("cuda")

    path_hdf5 = f"output/image-captioner/{model_name}-{dataset_name}-{split}.h5"

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

            inputs = processor(raw_image, return_tensors="pt")
            inputs = inputs.to("cuda")
            out = model.generate(**inputs, **config["generate-kwargs"])
            generated_captions = processor.batch_decode(out, skip_special_tokens=True)

            # image_features = model.vision_model(**inputs)[0]
            # image_features = image_features.detach().cpu().numpy()

            # group.create_dataset("image-features", data=image_features)
            group.create_dataset("generated-captions", data=generated_captions)


if __name__ == "__main__":
    main()
