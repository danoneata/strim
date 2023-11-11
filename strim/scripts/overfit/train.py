import pdb

from functools import partial
from pathlib import Path

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

from strim.data import Flickr8kDataset
from strim.audio_to_image.train import (
    AudioToImageMapper,
    get_sample,
    H5_PATH_AUDIO,
    H5_PATH_IMAGE,
    MODELS,
)


NUM_BATCHES = 10
BATCH_SIZE = 32
OUT_DIR = Path("output/audio-to-image-mapper-overfit")


def get_datapipe(dataset_name, use_as, audio_h5, image_h5):
    assert dataset_name == "flickr8k", "Only Flickr8k is supported for now"
    dataset = Flickr8kDataset(split="train")
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

    datapipe = SequenceWrapper(range(BATCH_SIZE * NUM_BATCHES))
    datapipe = datapipe.map(get_sample_1)

    if use_as == "train":
        datapipe = datapipe.shuffle()
        datapipe = datapipe.cycle()
    else:
        datapipe = datapipe.to_iter_datapipe()

    datapipe = datapipe.batch(BATCH_SIZE)
    datapipe = datapipe.collate(collate_fn)

    return datapipe


@click.command()
@click.option("-i", "--image-model", "image_model_name", default="blip-base")
@click.option("-a", "--audio-model", "audio_model_name", default="wav2vec2-xls-r-2b")
@click.option("-d", "--dataset", "dataset_name", default="flickr8k")
@click.option("-m", "--mapping-model", "mapping_model_name")
def main(image_model_name, audio_model_name, dataset_name, mapping_model_name):
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
        h5py.File(H5_PATH_AUDIO.format(audio_model_name, dataset_name, "train"), "r"),
        h5py.File(H5_PATH_IMAGE.format(image_model_name, dataset_name, "train"), "r"),
    )

    lr = 3e-4
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

    # num_epochs_warmup = 5
    # num_steps = num_epochs * num_steps_per_epoch
    # warmup_duration = num_epochs_warmup * num_steps_per_epoch

    # torch_lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_steps)
    # scheduler = create_lr_scheduler_with_warmup(
    #     torch_lr_scheduler,
    #     warmup_start_value=1e-6,
    #     warmup_end_value=lr,
    #     warmup_duration=warmup_duration,
    # )

    # trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

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
        OUTDIR / f"{mapping_model_name}-{dataset_name}-{audio_model_name}-{image_model_name}",
        n_saved=None,
        filename_prefix="best",
        score_function=score_function,
        score_name="neg-loss",
        global_step_transform=global_step_from_engine(trainer),
    )

    evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})
    trainer.run(train_loader, max_epochs=num_epochs, epoch_length=num_steps_per_epoch)


if __name__ == "__main__":
    main()
