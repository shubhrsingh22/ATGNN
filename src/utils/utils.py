import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

from omegaconf import DictConfig
from pytorch_lightning import  LightningModule
from src.utils import pylogger, rich_utils
import torch 
import torch.nn.functional as F

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

def load_weights(model: LightningModule, cfg: Dict, pretrained: str):
    """Initialise the PGN encoder from a checkpoint.

    `pretrained='img'` expects an ImageNet Pyramid ViG checkpoint
    (https://github.com/huawei-noah/Efficient-AI-Backbones, e.g. pvig_s_82.1.pth.tar).
    The 3-channel stem is averaged to 1 channel, positional/relative embeddings are
    interpolated to the spectrogram grid, the ImageNet prediction head is skipped and
    ATGNN-specific parameters (label embeddings, MLG blocks) keep their random init.
    """
    ckpt = torch.load(cfg.get('pretrain_path'), map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt

    if pretrained == "audioset":
        model.net.load_state_dict(state_dict)
        return

    own_state = model.net.state_dict()
    loaded, skipped = 0, []
    for name, param in state_dict.items():
        if name not in own_state:
            skipped.append(name)
            continue

        # Average across input channels of the ImageNet stem to get a single channel
        if name == 'stem.convs.0.weight':
            param = torch.mean(param, dim=1, keepdim=True)

        # Interpolate positional embedding from the 224x224 image grid to the mel grid
        if name == 'pos_embed':
            param = F.interpolate(
                param, size=own_state[name].shape[-2:], mode='bicubic', align_corners=False)

        # Interpolate relative position tables to the new node counts
        if 'relative_pos' in name:
            h, w = own_state[name].shape[-2:]
            param = F.interpolate(
                param.unsqueeze(1), size=(h, w), mode='bicubic', align_corners=False).squeeze(1)

        # ImageNet 1000-class head is not transferable
        if 'prediction' in name:
            continue

        if param.shape != own_state[name].shape:
            skipped.append(name)
            continue
        own_state[name].copy_(param)
        loaded += 1

    model.net.load_state_dict(own_state)
    log.info(f"Loaded {loaded} tensors from {cfg.get('pretrain_path')}; skipped {len(skipped)}")
        


    
def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
