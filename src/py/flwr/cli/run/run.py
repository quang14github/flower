# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower command line interface `run` command."""

import sys

import typer

from flwr.cli import config_utils
from flwr.simulation.run_simulation import _run_simulation


def run() -> None:
    """Run Flower project."""
    typer.secho("Loading project configuration... ", fg=typer.colors.BLUE)

    config, errors, warnings = config_utils.load_and_validate_with_defaults()

    if config is None:
        typer.secho(
            "Project configuration could not be loaded.\npyproject.toml is invalid:\n"
            + "\n".join([f"- {line}" for line in errors]),
            fg=typer.colors.RED,
            bold=True,
        )
        sys.exit()

    if warnings:
        typer.secho(
            "Project configuration is missing the following "
            "recommended properties:\n" + "\n".join([f"- {line}" for line in warnings]),
            fg=typer.colors.RED,
            bold=True,
        )

    typer.secho("Success", fg=typer.colors.GREEN)

    server_app_ref = config["flower"]["components"]["serverapp"]
    client_app_ref = config["flower"]["components"]["clientapp"]
    engine = config["flower"]["engine"]["name"]

    if engine == "simulation":
        num_supernodes = config["flower"]["engine"]["simulation"]["supernode"]["num"]

        typer.secho("Starting run... ", fg=typer.colors.BLUE)
        _run_simulation(
            server_app_attr=server_app_ref,
            client_app_attr=client_app_ref,
            num_supernodes=num_supernodes,
        )
    else:
        typer.secho(
            f"Engine '{engine}' is not yet supported in `flwr run`",
            fg=typer.colors.RED,
            bold=True,
        )
