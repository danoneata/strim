from collections import OrderedDict

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR  # type: ignore

from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *

# create default evaluator for doctests


def eval_step(engine, batch):
    return batch


default_evaluator = Engine(eval_step)

# create default optimizer for doctests

lr = 4e-3
param_tensor = torch.zeros([1], requires_grad=True)
optimizer = torch.optim.SGD([param_tensor], lr=lr)

# create default trainer for doctests
# as handlers could be attached to the trainer,
# each test must define his own trainer using `.. testsetup:`


def get_default_trainer():
    def train_step(engine, batch):
        return batch

    return Engine(train_step)


# create default model for doctests

default_model = nn.Sequential(
    OrderedDict([("base", nn.Linear(4, 2)), ("fc", nn.Linear(2, 1))])
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

default_trainer = get_default_trainer()
default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)


@default_trainer.on(Events.ITERATION_COMPLETED)
def print_lr():
    print(
        "{:6d} · {:.5f}".format(
            default_trainer.state.iteration,
            optimizer.param_groups[0]["lr"],
        )
    )


default_trainer.run([0] * 8, max_epochs=num_epochs, epoch_length=num_steps_per_epoch)
