from typing import Dict, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

import pinnstorch


def read_data_fn(root_path):
    data = pinnstorch.utils.load_data(root_path, "burgers_shock.mat")
    exact_u = np.real(data["usol"])
    return {"u": exact_u}


def pde_fn(outputs: Dict[str, torch.Tensor],
           x: torch.Tensor,
           t: torch.Tensor):
    u_x, u_t = pinnstorch.utils.gradient(outputs["u"], [x, t])
    u_xx = pinnstorch.utils.gradient(u_x, x)[0]
    outputs["f"] = u_t + outputs["u"] * u_x - (0.01 / np.pi) * u_xx
    return outputs


@hydra.main(version_base="1.3", config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    pinnstorch.utils.extras(cfg)

    metric_dict, _ = pinnstorch.train(
        cfg, read_data_fn=read_data_fn, pde_fn=pde_fn, output_fn=None
    )

    metric_value = pinnstorch.utils.get_metric_value(
        metric_dict=metric_dict, metric_names=cfg.get("optimized_metric")
    )
    return metric_value


if __name__ == "__main__":
    main()
