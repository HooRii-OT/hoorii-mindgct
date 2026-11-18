# Copyright 2026 HooRii Technology
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
"""Shared state of the x86 path and folding of low-rank adapters into dense weights."""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F


STATE = {'on': False}


def effective(module, adapters=()):
    dense = module.base if hasattr(module, 'base') else module
    weight = dense.weight.detach().float().clone()
    terms = ([module] if hasattr(module, 'base') and hasattr(module, 'A') else []) + list(adapters)
    with torch.no_grad():
        for lo in terms:
            weight.add_(lo.B.detach().float() @ lo.A.detach().float(), alpha=lo.scale)
    assert getattr(dense, 'bias', None) is None
    return weight


