from types import SimpleNamespace

import pytest
import torch

from models import networks
from models.base_model import BaseModel
import util.visualizer as visualizer_module


def test_get_scheduler_invalid_policy_raises():
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([param], lr=0.01)
    opt = SimpleNamespace(lr_policy="invalid", epoch_count=1, n_epochs=1, n_epochs_decay=1, lr_decay_iters=1)

    with pytest.raises(NotImplementedError, match=r"learning rate policy \[invalid\] is not implemented"):
        networks.get_scheduler(optimizer, opt)


class DummyModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.model_names = ["G"]
        self.netG = torch.nn.Conv2d(3, 3, kernel_size=1)

    def set_input(self, input_data):
        pass

    def forward(self):
        pass

    def optimize_parameters(self):
        pass


def test_ddp_rejects_plain_batch_norm(monkeypatch, tmp_path):
    monkeypatch.setattr("models.base_model.networks.init_net", lambda net, init_type, init_gain: net)
    monkeypatch.setattr("models.base_model.dist.is_initialized", lambda: True)

    opt = SimpleNamespace(
        isTrain=True,
        checkpoints_dir=str(tmp_path),
        name="ddp_norm_guard",
        device=torch.device("cpu"),
        preprocess="resize_and_crop",
        init_type="normal",
        init_gain=0.02,
        continue_train=False,
        norm="batch",
        verbose=False,
    )

    model = DummyModel(opt)
    with pytest.raises(ValueError, match="--norm batch is not supported"):
        model.setup(opt)


def _make_visualizer_opt(tmp_path, use_wandb):
    checkpoints_dir = tmp_path / "checkpoints"
    (checkpoints_dir / "exp").mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        isTrain=True,
        no_html=True,
        display_winsize=256,
        name="exp",
        use_wandb=use_wandb,
        checkpoints_dir=str(checkpoints_dir),
        wandb_project_name="test",
    )


def test_visualizer_allows_disabled_wandb_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(visualizer_module, "wandb", None)
    opt = _make_visualizer_opt(tmp_path, use_wandb=False)
    visualizer = visualizer_module.Visualizer(opt)
    assert visualizer.use_wandb is False


def test_visualizer_raises_when_wandb_requested_but_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(visualizer_module, "wandb", None)
    opt = _make_visualizer_opt(tmp_path, use_wandb=True)
    with pytest.raises(ImportError, match="wandb package cannot be found"):
        visualizer_module.Visualizer(opt)
