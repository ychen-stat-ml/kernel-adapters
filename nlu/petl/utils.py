from collections import OrderedDict
import torch, math
from torch import nn
import numpy as np

from typing import Sequence, Tuple

aggregators = OrderedDict()

# copie from https://github.com/pytorch/fairseq/blob/14c5bd027f04aae9dbb32f1bd7b34591b61af97f/fairseq/tasks/online_backtranslation.py#L41
class PiecewiseLinearFn:
    """Piecewise linear function. Can be configured with a string."""

    def __init__(self, pieces: Sequence[Tuple[int, float]]):
        assert pieces == sorted(
            pieces
        ), f"PiecewiseLinearFn configuration should be sorted, received: {pieces}"

        self.pieces = pieces

    def __call__(self, x: int) -> float:
        for i, (x_a, y_a) in enumerate(self.pieces[:-1]):
            x_b, y_b = self.pieces[i + 1]
            if x_a <= x <= x_b:
                return y_a + (x - x_a) * (y_b - y_a) / (x_b - x_a)

        return self.pieces[-1][1]

    @staticmethod
    def from_string(configuration: str) -> "PiecewiseLinearFn":
        """
        Parse the configuration of lambda coefficient (for scheduling).
        x = "3"                  # lambda will be a constant equal to x
        x = "0:1,1000:0"         # lambda will start from 1 and linearly decrease
                                 # to 0 during the first 1000 iterations
        x = "0:0,1000:0,2000:1"  # lambda will be equal to 0 for the first 1000
                                 # iterations, then will linearly increase to 1 until iteration 2000
        """
        if isinstance(configuration, float):
            return PiecewiseLinearFn([(0, configuration)])

        try:
            parts = configuration.split(",")
            if len(parts) == 1:
                v = float(configuration)
                return PiecewiseLinearFn([(0, v)])

            split = [s.split(":") for s in parts]
            pieces = [(int(t), float(v)) for t, v in split]
            return PiecewiseLinearFn(pieces)
        except Exception:
            raise ValueError(
                f"Invalid PiecewiseLinearFn configuration: {configuration!r}"
            )

    @staticmethod
    def one() -> "PiecewiseLinearFn":
        return PiecewiseLinearFn([(0, 1.0)])

def safe_round(number, ndigits):
    if hasattr(number, "__round__"):
        return round(number, ndigits)
    elif torch is not None and torch.is_tensor(number) and number.numel() == 1:
        return safe_round(number.item(), ndigits)
    elif np is not None and np.ndim(number) == 0 and hasattr(number, "item"):
        return safe_round(number.item(), ndigits)
    else:
        return number


class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self, round: int = None):
        self.round = round
        self.reset()

    def reset(self):
        self.val = None  # most recent update
        self.sum = 0  # sum from all updates
        self.count = 0  # total n from all updates

    def update(self, val, n=1):
        if val is not None:
            self.val = val
            if n > 0:
                self.sum = self.sum + (val * n)
                self.count = self.count + n

    def state_dict(self):
        return {
            "val": self.val,
            "sum": self.sum,
            "count": self.count,
            "round": self.round,
        }

    def load_state_dict(self, state_dict):
        self.val = state_dict["val"]
        self.sum = state_dict["sum"]
        self.count = state_dict["count"]
        self.round = state_dict.get("round", None)

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else self.val

    @property
    def smoothed_value(self) -> float:
        val = self.avg
        if self.round is not None and val is not None:
            val = safe_round(val, self.round)
        return val


def log_metrics(key, value, weight):
    if key not in aggregators:
        aggregators[key] = AverageMeter(round=7)
    aggregators[key].update(value, weight)


def get_aggerators():
    return aggregators

from transformers.trainer_pt_utils import get_parameter_names
from transformers.optimization import get_scheduler
from transformers import Trainer

def convert_str_to_list(s):
    if s=="":
        return []
    else:
        return [int(item) for item in s.split(',')]

def index_list_in_str(idxstr, s):
    res = [i in s for i in idxstr.split(',')]
    return sum(res) == 1

# TODO: different lr for sketched layers
def create_optimizer(args, opt_model, lr_sketch_factor=1, idxstr=None):
    """
    Setup the optimizer.

    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through `optimizers`, or subclass and override this method in a subclass.
    """
    # opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

    decay_parameters = get_parameter_names(opt_model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    # print(decay_parameters)
    
    # TODO: better if classifier also has a larger lr?
    sketch_parameters = get_parameter_names(opt_model, [nn.LayerNorm])
    sketch_parameters = [name for name in sketch_parameters 
        if "attention" not in name and "dense" in name 
        and index_list_in_str(idxstr, name)]
    print(sketch_parameters)

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in opt_model.named_parameters() 
                if n in decay_parameters and n in sketch_parameters],
            "weight_decay": args.weight_decay,
            "lr": lr_sketch_factor * args.learning_rate
        },
        {
            "params": [p for n, p in opt_model.named_parameters() 
                if n not in decay_parameters and n in sketch_parameters],
            "weight_decay": 0.0,
            "lr": lr_sketch_factor * args.learning_rate
        },
        {
            "params": [p for n, p in opt_model.named_parameters() 
                if n in decay_parameters and n not in sketch_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in opt_model.named_parameters() 
                if n not in decay_parameters and n not in sketch_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return optimizer

def create_scheduler(args, len_dataset, 
        optimizer: torch.optim.Optimizer = None):
    """
    Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
    passed as an argument.

    Args:
        num_training_steps (int): The number of training steps to do.
    """
    len_dataloader = math.ceil(len_dataset / args.per_device_train_batch_size)
    num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    num_training_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)

    # print(num_training_steps, args.get_warmup_steps(num_training_steps))
    # 1145, 69 -> should be 1150

    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.get_warmup_steps(num_training_steps),
        num_training_steps=num_training_steps,
    )
    return lr_scheduler