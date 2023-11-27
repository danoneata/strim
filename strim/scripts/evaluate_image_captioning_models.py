import pandas as pd
from strim.data import Flickr8kDataset
from strim.scripts.show_bleu_metrics import compute_score_image_captions, GeneratedCaptionsLoader


split = "test"
dataset = Flickr8kDataset(split=split)

num_refs = 5
num_repeats = 5

model_names = ["blip-base", "blip-large", "blip2-opt-2.7b", "git-base-coco"]
generation_types = ["topk", "sample", "diverse"]


load_generated_captions = {
    m + "-" + g: GeneratedCaptionsLoader(m + "-" + g, "flickr8k", split)
    for m in model_names
    for g in generation_types
}

data = [
    {
        "model": m,
        "generation": g,
        "score": compute_score_image_captions(dataset, load_generated_captions[m + "-" + g], num_refs),
        "num": r,
    }
    for m in model_names
    for g in generation_types
    for r in range(num_repeats)
]

df = pd.DataFrame(data)
df = df.groupby(["model", "generation"])["score"].agg(["mean", "std"])
df = df.reset_index()
df["mean-std"] = df["mean"].map("{:.2f}".format) + "±" + df["std"].map("{:.1f}".format)
df = df.drop(columns=["mean", "std"])
df = df.pivot(index="model", columns="generation", values="mean-std")
df = df[generation_types]
print(df)
print(df.to_csv(sep=",", index=False))
