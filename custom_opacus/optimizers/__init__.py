# Copyright (c) Meta Platforms, Inc. and affiliates.
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

from .ddp_perlayeroptimizer import DistributedPerLayerOptimizer
from .ddpoptimizer import DistributedDPOptimizer
from .optimizer import DPOptimizer
from .perlayeroptimizer import DPPerLayerOptimizer
from .topk_influence_optimizer import SparsefluenceOptimizer


__all__ = [
    "DistributedPerLayerOptimizer",
    "DistributedDPOptimizer",
    "DPOptimizer",
    "DPPerLayerOptimizer",
    "DPOptimizerTopk",
    "InfluenceBoundedDPOptimizer",
    "SparsefluenceOptimizer"
]


def get_optimizer_class(clipping: str, distributed: bool):
    if clipping == "flat" and distributed is False:
        return DPOptimizer
    elif clipping == "flat" and distributed is True:
        return DistributedDPOptimizer
    elif clipping == "per_layer" and distributed is False:
        return DPPerLayerOptimizer
    elif clipping == "per_layer" and distributed is True:
        return DistributedPerLayerOptimizer
    elif clipping == "budget":
        return SparsefluenceOptimizer

    raise ValueError(
        f"Unexpected optimizer parameters. Clipping: {clipping}, distributed: {distributed}"
    )
