"""Runs CNN federated learning for MNIST dataset."""

from pathlib import Path

import flwr as fl
import hydra
import numpy as np
import torch
from omegaconf import DictConfig

import client, utils

cuda = torch.cuda.is_available()
if cuda:
    cuda_device = torch.cuda.current_device()
    print(f"Using CUDA device {cuda_device}")
DEVICE: torch.device = torch.device(f"cuda:{cuda_device}" if cuda else "cpu")


@hydra.main(config_path="docs/conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function to run CNN federated learning on MNIST.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    client_fn, testloader = client.gen_client_fn(
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        device=DEVICE,
        num_clients=cfg.num_clients,
        iid=cfg.iid,
        balance=cfg.balance,
        learning_rate=cfg.learning_rate,
    )

    evaluate_fn = utils.gen_evaluate_fn(testloader, DEVICE)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=cfg.client_fraction,
        fraction_evaluate=1.0,
        min_fit_clients=int(cfg.num_clients * cfg.client_fraction),
        min_evaluate_clients=cfg.num_clients,
        min_available_clients=cfg.num_clients,
        evaluate_fn=evaluate_fn,
        evaluate_metrics_aggregation_fn=utils.weighted_average,
    )

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
    )

    file_suffix: str = (
        f"{'_iid' if cfg.iid else ''}"
        f"{'_balanced' if cfg.balance else ''}"
        f"_C={cfg.num_clients}"
        f"_B={cfg.batch_size}"
        f"_E={cfg.num_epochs}"
        f"_R={cfg.num_rounds}"
    )

    np.save(
        Path(cfg.save_path) / Path(f"hist{file_suffix}"),
        history,  # type: ignore
    )

    utils.plot_metric_from_history(
        history,
        cfg.save_path,
        cfg.expected_maximum,
        file_suffix,
    )


if __name__ == "__main__":
    main()
