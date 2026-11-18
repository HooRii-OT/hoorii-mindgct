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
"""Single-row Linear through a pre-transposed weight, checked bitwise at install."""
import types


def install(model, enabled=True):
    import torch
    from torch.nn import functional as F
    skip = {id(m) for m in model.lm.modules()} | {id(m) for m in model.rca.modules()}
    state = {'enabled': enabled, 'modules': [], 'calls': 0}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or id(module) in skip or name.startswith('lm.') or name.startswith('rca.'):
            continue
        w = module.weight.detach()
        if w.dtype != torch.float32 or w.device.type != 'cpu':
            continue
        wt = w.t().contiguous()
        b = module.bias.detach() if module.bias is not None else None
        probe = torch.randn(1, w.shape[1])
        ref = F.linear(probe, w, b)
        alt = torch.addmm(b, probe, wt) if b is not None else probe @ wt
        if not torch.equal(ref, alt):
            continue
        original = module.forward

        def forward(self, x, _wt=wt, _b=b, _orig=original):
            if state['enabled'] and x.dim() >= 1 and x.numel() == x.shape[-1] and x.dtype == torch.float32 and not torch.is_grad_enabled():
                state['calls'] += 1
                flat = x.reshape(1, -1)
                y = torch.addmm(_b, flat, _wt) if _b is not None else flat @ _wt
                return y.reshape(*x.shape[:-1], _wt.shape[1])
            return _orig(x)
        module.forward = types.MethodType(forward, module)
        state['modules'].append(name)
    return state
