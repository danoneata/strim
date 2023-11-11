import pdb

from functools import partial
import click
import h5py  # type: ignore
import torch  # type: ignore

from torchdata.datapipes.map import SequenceWrapper  # type: ignore
from torch.optim.lr_scheduler import CosineAnnealingLR  # type: ignore
from torch.nn.utils.rnn import pad_sequence  # type: ignore

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator  # type: ignore
from ignite.handlers import create_lr_scheduler_with_warmup, ModelCheckpoint, global_step_from_engine  # type: ignore
from ignite.metrics import Loss  # type: ignore
from ignite.utils import convert_tensor

import ignite.distributed as idist

from strim.data import Flickr8kDataset
from strim.scripts.extract_audio_features import get_group_name


H5_PATH_AUDIO = "output/audio-features/{}-{}-{}.h5"
H5_PATH_IMAGE = "output/image-captioner/{}-{}-{}.h5"



class AudioToImageMapper(torch.nn.Module):
    def __init__(self, dim_audio, dim_image, len_image_seq, **kwargs):
        super().__init__()
        self.transformer = torch.nn.Transformer(
            d_model=dim_image,
            batch_first=True,
            **kwargs,
        )
        self.queries = torch.nn.Parameter(torch.randn(1, len_image_seq, dim_image))
        self.projection = torch.nn.Linear(dim_audio, dim_image)

    def forward(self, input):
        audio_feat, padding_mask = input
        audio_feat = self.projection(audio_feat)
        B, _, _ = audio_feat.shape
        queries = self.queries.repeat(B, 1, 1)
        output = self.transformer(
            src=audio_feat,
            tgt=queries,
            src_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask,
        )
        return output


def get_sample(dataset, audio_h5, image_h5, i):
    max_audio_len = 600
    sample = dataset[i]
    path_audio = get_group_name(sample) + "/" + "audio-features"
    path_image = sample["key-image"] + "/" + "image-features"
    audio_feat = audio_h5[path_audio][...]
    audio_feat = audio_feat[:max_audio_len]
    image_feat = image_h5[path_image][...]
    return {
        "audio-feat": torch.tensor(audio_feat),
        "image-feat": torch.tensor(image_feat),
    }


def get_datapipe(dataset_name, split, audio_h5, image_h5):
    assert dataset_name == "flickr8k", "Only Flickr8k is supported for now"
    dataset = Flickr8kDataset(split=split)
    batch_size = 32
    # Some audio files are very long. Truncate them to avoid out-of-memory errors.

    def collate_fn(batch):
        audio_feats = [sample["audio-feat"] for sample in batch]
        audio_feats = pad_sequence(audio_feats, batch_first=True)
        image_feats = [sample["image-feat"] for sample in batch]
        image_feats = torch.cat(image_feats, dim=0)
        padding_mask = [
            torch.full((sample["audio-feat"].shape[0],), fill_value=False)
            for sample in batch
        ]
        padding_mask = pad_sequence(padding_mask, batch_first=True, padding_value=True)
        return {
            "audio-feat": audio_feats,
            "image-feat": image_feats,
            "padding-mask": padding_mask,
        }

    get_sample_1 = partial(get_sample, dataset, audio_h5, image_h5)

    # datapipe = SequenceWrapper(range(32 * 10))
    datapipe = SequenceWrapper(range(len(dataset)))
    datapipe = datapipe.map(get_sample_1)

    if split == "train":
        datapipe = datapipe.shuffle()
        datapipe = datapipe.cycle()
    else:
        datapipe = datapipe.to_iter_datapipe()
        datapipe = datapipe.header(batch_size * 16)

    datapipe = datapipe.batch(batch_size)
    datapipe = datapipe.collate(collate_fn)

    return datapipe


MODELS = {
    "tiny": {
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_feedforward": 128,
    },
    "medium": {
        "num_encoder_layers": 1,
        "num_decoder_layers": 2,
        "dim_feedforward": 512,
    },
    "large": {
        "num_encoder_layers": 1,
        "num_decoder_layers": 4,
        "dim_feedforward": 768,
    },
}


def training(local_rank, image_model_name, audio_model_name, dataset_name, mapping_model_name):
    rank = idist.get_rank()

    device = "cuda"
    model = AudioToImageMapper(
        dim_audio=1920,
        dim_image=768,
        len_image_seq=577,
        **MODELS[mapping_model_name],
    )
    model = model.to(device)

    train_loader = get_datapipe(
        dataset_name,
        "train",
        h5py.File(H5_PATH_AUDIO.format(audio_model_name, dataset_name, "train"), "r"),
        h5py.File(H5_PATH_IMAGE.format(image_model_name, dataset_name, "train"), "r"),
    )
    valid_loader = get_datapipe(
        dataset_name,
        "dev",
        h5py.File(H5_PATH_AUDIO.format(audio_model_name, dataset_name, "dev"), "r"),
        h5py.File(H5_PATH_IMAGE.format(image_model_name, dataset_name, "dev"), "r"),
    )

    lr = 4e-3
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    metrics = {"loss": Loss(criterion)}

    def prepare_batch(batch, device, non_blocking):
        audio_feat = convert_tensor(batch["audio-feat"], device, non_blocking)
        image_feat = convert_tensor(batch["image-feat"], device, non_blocking)
        padding_mask = convert_tensor(batch["padding-mask"], device, non_blocking)
        x = audio_feat, padding_mask
        y = image_feat
        return x, y

    trainer = create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=criterion,
        device=device,
        prepare_batch=prepare_batch,
    )
    evaluator = create_supervised_evaluator(
        model,
        metrics=metrics,
        device=device,
        prepare_batch=prepare_batch,
    )

    log_interval = 10
    num_steps_per_epoch = 500
    num_epochs = 100
    num_epochs_warmup = 5
    num_steps = num_epochs * num_steps_per_epoch
    warmup_duration = num_epochs_warmup * num_steps_per_epoch

    torch_lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_steps)
    scheduler = create_lr_scheduler_with_warmup(
        torch_lr_scheduler,
        warmup_start_value=1e-6,
        warmup_end_value=lr,
        warmup_duration=warmup_duration,
    )

    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    if rank == 0:
        @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
        def log_training_loss(engine):
            print(
                "{:3d} · {:6d} ◇ {:.5f} · lr = {:.5f}".format(
                    engine.state.epoch,
                    engine.state.iteration,
                    engine.state.output,
                    optimizer.param_groups[0]["lr"],
                )
            )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(valid_loader)
        metrics = evaluator.state.metrics
        losses_str = " · ".join(f"{k} = {v:.5f}" for k, v in metrics.items())
        print(
            "{:3d} · {:6s} ◇ {:s}".format(
                trainer.state.epoch,
                "valid",
                losses_str,
            )
        )

    def score_function(engine):
        return -engine.state.metrics["loss"]

    model_checkpoint = ModelCheckpoint(
        f"output/audio-to-image-mapper/{mapping_model_name}-{dataset_name}-{audio_model_name}-{image_model_name}",
        n_saved=5,
        filename_prefix="best",
        score_function=score_function,
        score_name="neg-loss",
        global_step_transform=global_step_from_engine(trainer),
    )

    evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})
    trainer.run(train_loader, max_epochs=num_epochs, epoch_length=num_steps_per_epoch)


@click.command()
@click.option("-i", "--image-model", "image_model_name", default="blip-base")
@click.option("-a", "--audio-model", "audio_model_name", default="wav2vec2-xls-r-2b")
@click.option("-d", "--dataset", "dataset_name", default="flickr8k")
@click.option("-m", "--mapping-model", "mapping_model_name")
def main(image_model_name, audio_model_name, dataset_name, mapping_model_name):
    backend = "nccl"
    nproc_per_node = 4
    with idist.Parallel(backend=backend, nproc_per_node=nproc_per_node) as parallel:
        parallel.run(training, image_model_name, audio_model_name, dataset_name, mapping_model_name)


if __name__ == "__main__":
    main()
