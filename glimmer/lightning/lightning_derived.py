# Copyright The PyTorch Lightning team.
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

# Modifications copyright (C) 2021 - 2026 Immanuel Weber
# derived from https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/callbacks/lr_monitor.py

from typing import Any


def get_scheduler_names(schedulers: list[Any]) -> list[str]:
    names = []
    for scheduler in schedulers:
        sch = scheduler.scheduler
        if scheduler.name is not None:
            name = scheduler.name
        else:
            opt_name = "lr-" + sch.optimizer.__class__.__name__ + "-" + sch.__class__.__name__
            i, name = 1, opt_name
            # Multiple scheduler of the same type
            while True:
                if name not in names:
                    break
                i, name = i + 1, f"{opt_name}-{i}"

        param_groups = sch.optimizer.param_groups
        if len(param_groups) != 1:
            for i in range(len(param_groups)):
                names.append(f"{name}/pg{i + 1}")
        else:
            names.append(name)
    return names


def get_lrs(schedulers: list[Any], scheduler_names: list[str], interval: str) -> dict[str, Any]:
    latest_stat: dict[str, Any] = {}

    for name, scheduler in zip(scheduler_names, schedulers):
        if scheduler.interval == interval or interval == "any":
            opt = scheduler.scheduler.optimizer
            param_groups = opt.param_groups
            for i, pg in enumerate(param_groups):
                suffix = f"/pg{i + 1}" if len(param_groups) > 1 else ""
                lr = {f"{name}{suffix}": pg.get("lr")}
                latest_stat.update(lr)
        else:
            print(f"warning: interval {scheduler.interval} not supported yet.")

    return latest_stat
