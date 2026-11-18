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
"""Fused CPU kernels for the x86 path: RMSNorm, row quantisation, W8A8 GEMM epilogues, RoPE and attention helpers.
One C++ extension, compiled on first use with the PyTorch toolchain (-O3 -march=native -fopenmp)."""
from __future__ import annotations


import torch

_EXT = None

_SRC = r"""
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <cstdint>

using at::Tensor;

static inline void rms_row(const float* x, const float* w, float* o, int64_t n, float eps) {
  float ss = 0.f;
  #pragma omp simd reduction(+:ss)
  for (int64_t k = 0; k < n; ++k) ss += x[k] * x[k];
  float inv = 1.f / std::sqrt(ss / (float)n + eps);
  #pragma omp simd
  for (int64_t k = 0; k < n; ++k) o[k] = w[k] * (x[k] * inv);
}

Tensor rms_norm(const Tensor& x, const Tensor& w, double eps) {
  TORCH_CHECK(x.scalar_type() == at::kFloat && w.scalar_type() == at::kFloat, "rms_norm fp32");
  auto xc = x.contiguous();
  const int64_t n = w.numel(); const int64_t rows = xc.numel() / n;
  auto out = at::empty_like(xc);
  const float* xp = xc.data_ptr<float>(); const float* wp = w.data_ptr<float>(); float* op = out.data_ptr<float>();
  at::parallel_for(0, rows, 16, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) rms_row(xp + i * n, wp, op + i * n, n, (float)eps);
  });
  return out;
}

std::vector<Tensor> quant_rows(const Tensor& x) {
  const int64_t M = x.size(0), K = x.size(1);
  auto q = at::empty({M, K}, x.options().dtype(at::kChar));
  auto s = at::empty({M}, x.options());
  const float* xp = x.data_ptr<float>(); int8_t* qp = q.data_ptr<int8_t>(); float* sp = s.data_ptr<float>();
  at::parallel_for(0, M, 8, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
      const float* r = xp + i * K; float m = 0.f;
      #pragma omp simd reduction(max:m)
      for (int64_t k = 0; k < K; ++k) { float a = std::fabs(r[k]); m = a > m ? a : m; }
      float sc = (m > 1e-20f ? m : 1e-20f) / 127.f; float inv = 1.f / sc; sp[i] = sc;
      int8_t* qr = qp + i * K;
      #pragma omp simd
      for (int64_t k = 0; k < K; ++k) {
        float v = std::nearbyint(r[k] * inv); v = v > 127.f ? 127.f : (v < -127.f ? -127.f : v); qr[k] = (int8_t)v;
      }
    }
  });
  return {q, s};
}

// out = acc * rs[i] * cs[j] (+ res)
Tensor w8a8(const Tensor& x, const Tensor& wqt, const Tensor& cs, const c10::optional<Tensor>& res) {
  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kFloat && x.is_contiguous(), "w8a8: fp32 contiguous 2-D");
  auto qs = quant_rows(x);
  Tensor q = qs[0]; const int64_t M = q.size(0);
  if (M < 17) q = at::cat({q, at::zeros({17 - M, q.size(1)}, q.options())}, 0);
  auto acc = at::_int_mm(q, wqt);
  const int64_t N = acc.size(1);
  auto out = at::empty({M, N}, x.options());
  const int32_t* ap = acc.data_ptr<int32_t>(); const float* r = qs[1].data_ptr<float>(); const float* c = cs.data_ptr<float>();
  float* op = out.data_ptr<float>();
  const float* rp = nullptr; Tensor rc;
  if (res.has_value()) { rc = res->contiguous(); rp = rc.data_ptr<float>(); }
  at::parallel_for(0, M, 4, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
      const int32_t* ar = ap + i * N; float* orow = op + i * N; const float ri = r[i];
      if (rp) { const float* rr = rp + i * N;
        #pragma omp simd
        for (int64_t j = 0; j < N; ++j) orow[j] = rr[j] + (float)ar[j] * (ri * c[j]);
      } else {
        #pragma omp simd
        for (int64_t j = 0; j < N; ++j) orow[j] = (float)ar[j] * (ri * c[j]);
      }
    }
  });
  return out;
}

Tensor silu_mul(const Tensor& gu, int64_t I) {
  const int64_t R = gu.size(0);
  auto out = at::empty({R, I}, gu.options());
  const float* g = gu.data_ptr<float>(); float* o = out.data_ptr<float>(); const int64_t W = gu.size(1);
  at::parallel_for(0, R, 4, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
      const float* a = g + i * W; const float* u = a + I; float* orow = o + i * I;
      #pragma omp simd
      for (int64_t k = 0; k < I; ++k) { float v = a[k]; orow[k] = (v / (1.f + std::exp(-v))) * u[k]; }
    }
  });
  return out;
}

static inline void norm_rope(const float* x, const float* w, const float* cs, const float* sn, float* o, int64_t hd, float eps) {
  float tmp[512];
  rms_row(x, w, tmp, hd, eps);
  const int64_t h = hd / 2;
  for (int64_t j = 0; j < h; ++j) {
    o[j] = tmp[j] * cs[j] - tmp[j + h] * sn[j];
    o[j + h] = tmp[j + h] * cs[j + h] + tmp[j] * sn[j + h];
  }
}

// qkv (R, (hq+2hkv)*hd) -> q_req (hkv, R, rep, hd), q_pad (M, hkv, T, rep, hd), k_pad / v_pad (M, hkv, T, hd)
std::vector<Tensor> qkv_prep(const Tensor& qkv, const Tensor& qw, const Tensor& kw, const Tensor& cosv, const Tensor& sinv,
                             double eps, int64_t hq, int64_t hkv, const Tensor& blk, const Tensor& tpos, int64_t M, int64_t T) {
  const int64_t R = qkv.size(0), hd = qw.numel(), rep = hq / hkv, W = qkv.size(1);
  TORCH_CHECK(hd <= 512 && qkv.is_contiguous() && cosv.is_contiguous() && sinv.is_contiguous(), "qkv_prep");
  auto opt = qkv.options();
  auto q_req = at::empty({hkv, R, rep, hd}, opt);
  auto q_pad = at::zeros({M, hkv, T, rep, hd}, opt);
  auto k_pad = at::zeros({M, hkv, T, hd}, opt);
  auto v_pad = at::zeros({M, hkv, T, hd}, opt);
  const float* X = qkv.data_ptr<float>(); const float* QW = qw.data_ptr<float>(); const float* KW = kw.data_ptr<float>();
  const float* C = cosv.data_ptr<float>(); const float* S = sinv.data_ptr<float>();
  const int64_t* B = blk.data_ptr<int64_t>(); const int64_t* TP = tpos.data_ptr<int64_t>();
  float* QR = q_req.data_ptr<float>(); float* QP = q_pad.data_ptr<float>(); float* KP = k_pad.data_ptr<float>(); float* VP = v_pad.data_ptr<float>();
  at::parallel_for(0, R, 4, [&](int64_t b0, int64_t e0) {
    for (int64_t r = b0; r < e0; ++r) {
      const float* x = X + r * W; const float* cs = C + r * hd; const float* sn = S + r * hd;
      const int64_t b = B[r], t = TP[r];
      for (int64_t hh = 0; hh < hq; ++hh) {
        const int64_t g = hh / rep, i = hh % rep;
        float* dst = QR + ((g * R + r) * rep + i) * hd;
        norm_rope(x + hh * hd, QW, cs, sn, dst, hd, (float)eps);
        float* dp = QP + ((((b * hkv + g) * T + t) * rep) + i) * hd;
        for (int64_t j = 0; j < hd; ++j) dp[j] = dst[j];
      }
      for (int64_t g = 0; g < hkv; ++g) {
        norm_rope(x + (hq + g) * hd, KW, cs, sn, KP + ((b * hkv + g) * T + t) * hd, hd, (float)eps);
        const float* vs = x + (hq + hkv + g) * hd; float* vd = VP + ((b * hkv + g) * T + t) * hd;
        for (int64_t j = 0; j < hd; ++j) vd[j] = vs[j];
      }
    }
  });
  return {q_req, q_pad, k_pad, v_pad};
}

// in place: s_req (hkv, R*rep, L) and s_own (M, hkv, T*rep, T) -> probabilities over [req keys | own causal keys]
void softmax2(Tensor& s_req, Tensor& s_own, const Tensor& req_ok, const Tensor& own_ok, const Tensor& blk, const Tensor& tpos,
              double scale, int64_t rep) {
  const int64_t hkv = s_req.size(0), RR = s_req.size(1), L = s_req.size(2), R = RR / rep;
  const int64_t T = s_own.size(3);
  float* SR = s_req.data_ptr<float>(); float* SO = s_own.data_ptr<float>();
  const bool* RQ = req_ok.data_ptr<bool>(); const bool* OK = own_ok.data_ptr<bool>();
  const int64_t* B = blk.data_ptr<int64_t>(); const int64_t* TP = tpos.data_ptr<int64_t>();
  const float sc = (float)scale;
  at::parallel_for(0, hkv * R, 8, [&](int64_t b0, int64_t e0) {
    for (int64_t idx = b0; idx < e0; ++idx) {
      const int64_t g = idx / R, r = idx % R, b = B[r], t = TP[r];
      const bool* ok = OK + b * T;
      for (int64_t i = 0; i < rep; ++i) {
        float* a = SR + (g * RR + r * rep + i) * L;
        float* o = SO + ((((b * hkv + g) * T + t) * rep) + i) * T;
        float mx = -INFINITY;
        for (int64_t k = 0; k < L; ++k) if (RQ[k]) { float v = a[k] * sc; mx = v > mx ? v : mx; }
        for (int64_t k = 0; k <= t; ++k) if (ok[k]) { float v = o[k] * sc; mx = v > mx ? v : mx; }
        float sum = 0.f;
        for (int64_t k = 0; k < L; ++k) { float v = RQ[k] ? std::exp(a[k] * sc - mx) : 0.f; a[k] = v; sum += v; }
        for (int64_t k = 0; k < T; ++k) { float v = (k <= t && ok[k]) ? std::exp(o[k] * sc - mx) : 0.f; o[k] = v; sum += v; }
        const float inv = 1.f / sum;
        #pragma omp simd
        for (int64_t k = 0; k < L; ++k) a[k] *= inv;
        #pragma omp simd
        for (int64_t k = 0; k < T; ++k) o[k] *= inv;
      }
    }
  });
}

// o (R, hq*hd) = o_req (hkv, R, rep, hd) + o_own (M, hkv, T, rep, hd)[blk, :, tpos]
Tensor combine(const Tensor& o_req, const Tensor& o_own, const Tensor& blk, const Tensor& tpos) {
  const int64_t hkv = o_req.size(0), R = o_req.size(1), rep = o_req.size(2), hd = o_req.size(3), T = o_own.size(2);
  auto out = at::empty({R, hkv * rep * hd}, o_req.options());
  const float* A = o_req.data_ptr<float>(); const float* Bo = o_own.data_ptr<float>(); float* O = out.data_ptr<float>();
  const int64_t* B = blk.data_ptr<int64_t>(); const int64_t* TP = tpos.data_ptr<int64_t>();
  at::parallel_for(0, R, 8, [&](int64_t b0, int64_t e0) {
    for (int64_t r = b0; r < e0; ++r) {
      const int64_t b = B[r], t = TP[r];
      for (int64_t g = 0; g < hkv; ++g) for (int64_t i = 0; i < rep; ++i) {
        const float* a = A + ((g * R + r) * rep + i) * hd;
        const float* c = Bo + ((((b * hkv + g) * T + t) * rep) + i) * hd;
        float* o = O + r * hkv * rep * hd + (g * rep + i) * hd;
        #pragma omp simd
        for (int64_t j = 0; j < hd; ++j) o[j] = a[j] + c[j];
      }
    }
  });
  return out;
}
// q/k-norm + RoPE: q_req (hkv, R, rep, hd); k_pad / v_pad (M, hkv, T, hd) (zero where no token)
std::vector<Tensor> qkv_prep2(const Tensor& qkv, const Tensor& qw, const Tensor& kw, const Tensor& cosv, const Tensor& sinv,
                              double eps, int64_t hq, int64_t hkv, const Tensor& blk, const Tensor& tpos, int64_t M, int64_t T) {
  const int64_t R = qkv.size(0), hd = qw.numel(), rep = hq / hkv, W = qkv.size(1);
  TORCH_CHECK(hd <= 512 && qkv.is_contiguous() && cosv.is_contiguous() && sinv.is_contiguous(), "qkv_prep2");
  auto opt = qkv.options();
  auto q_req = at::empty({hkv, R, rep, hd}, opt);
  auto k_pad = at::zeros({M, hkv, T, hd}, opt);
  auto v_pad = at::zeros({M, hkv, T, hd}, opt);
  const float* X = qkv.data_ptr<float>(); const float* QW = qw.data_ptr<float>(); const float* KW = kw.data_ptr<float>();
  const float* C = cosv.data_ptr<float>(); const float* S = sinv.data_ptr<float>();
  const int64_t* B = blk.data_ptr<int64_t>(); const int64_t* TP = tpos.data_ptr<int64_t>();
  float* QR = q_req.data_ptr<float>(); float* KP = k_pad.data_ptr<float>(); float* VP = v_pad.data_ptr<float>();
  at::parallel_for(0, R, 4, [&](int64_t b0, int64_t e0) {
    for (int64_t r = b0; r < e0; ++r) {
      const float* x = X + r * W; const float* cs = C + r * hd; const float* sn = S + r * hd;
      const int64_t b = B[r], t = TP[r];
      for (int64_t hh = 0; hh < hq; ++hh) {
        const int64_t g = hh / rep, i = hh % rep;
        norm_rope(x + hh * hd, QW, cs, sn, QR + ((g * R + r) * rep + i) * hd, hd, (float)eps);
      }
      for (int64_t g = 0; g < hkv; ++g) {
        norm_rope(x + (hq + g) * hd, KW, cs, sn, KP + ((b * hkv + g) * T + t) * hd, hd, (float)eps);
        const float* vs = x + (hq + hkv + g) * hd; float* vd = VP + ((b * hkv + g) * T + t) * hd;
        for (int64_t j = 0; j < hd; ++j) vd[j] = vs[j];
      }
    }
  });
  return {q_req, k_pad, v_pad};
}

// s_req (hkv, R*rep, L) raw scores -> in place probabilities of the request keys; returns o (R, hkv*rep*hd) = own-block part
// of the attention output (own causal keys of the row's block, scored here); the request part is added by add_req.
Tensor attn_own(Tensor& s_req, const Tensor& q_req, const Tensor& k_pad, const Tensor& v_pad, const Tensor& req_ok,
                const Tensor& own_ok, const Tensor& blk, const Tensor& tpos, double scale) {
  const int64_t hkv = q_req.size(0), R = q_req.size(1), rep = q_req.size(2), hd = q_req.size(3);
  const int64_t L = s_req.size(2), T = k_pad.size(2);
  auto out = at::empty({R, hkv * rep * hd}, q_req.options());
  float* SR = s_req.data_ptr<float>(); const float* Q = q_req.data_ptr<float>();
  const float* KP = k_pad.data_ptr<float>(); const float* VP = v_pad.data_ptr<float>(); float* O = out.data_ptr<float>();
  const bool* RQ = req_ok.data_ptr<bool>(); const bool* OK = own_ok.data_ptr<bool>();
  const int64_t* B = blk.data_ptr<int64_t>(); const int64_t* TP = tpos.data_ptr<int64_t>();
  const float sc = (float)scale;
  at::parallel_for(0, R * hkv, 8, [&](int64_t b0, int64_t e0) {
    float own[512];
    for (int64_t idx = b0; idx < e0; ++idx) {
      const int64_t r = idx / hkv, g = idx % hkv, b = B[r], t = TP[r];
      const bool* ok = OK + b * T;
      const float* kb = KP + (b * hkv + g) * T * hd; const float* vb = VP + (b * hkv + g) * T * hd;
      for (int64_t i = 0; i < rep; ++i) {
        const float* q = Q + ((g * R + r) * rep + i) * hd;
        float* a = SR + (g * R * rep + r * rep + i) * L;
        float mx = -INFINITY;
        for (int64_t k = 0; k < L; ++k) if (RQ[k]) { float v = a[k] * sc; mx = v > mx ? v : mx; }
        for (int64_t k = 0; k <= t; ++k) {
          if (!ok[k]) { own[k] = -INFINITY; continue; }
          const float* kk = kb + k * hd; float d = 0.f;
          #pragma omp simd reduction(+:d)
          for (int64_t j = 0; j < hd; ++j) d += q[j] * kk[j];
          own[k] = d * sc; mx = own[k] > mx ? own[k] : mx;
        }
        float sum = 0.f;
        for (int64_t k = 0; k < L; ++k) { float v = RQ[k] ? std::exp(a[k] * sc - mx) : 0.f; a[k] = v; sum += v; }
        for (int64_t k = 0; k <= t; ++k) { float v = ok[k] ? std::exp(own[k] - mx) : 0.f; own[k] = v; sum += v; }
        const float inv = 1.f / sum;
        #pragma omp simd
        for (int64_t k = 0; k < L; ++k) a[k] *= inv;
        float* o = O + r * hkv * rep * hd + (g * rep + i) * hd;
        #pragma omp simd
        for (int64_t j = 0; j < hd; ++j) o[j] = 0.f;
        for (int64_t k = 0; k <= t; ++k) {
          const float p = own[k] * inv; if (p == 0.f) continue;
          const float* vv = vb + k * hd;
          #pragma omp simd
          for (int64_t j = 0; j < hd; ++j) o[j] += p * vv[j];
        }
      }
    }
  });
  return out;
}

// o (R, hkv*rep*hd) += o_req (hkv, R, rep, hd)
void add_req(Tensor& o, const Tensor& o_req) {
  const int64_t hkv = o_req.size(0), R = o_req.size(1), rep = o_req.size(2), hd = o_req.size(3);
  float* O = o.data_ptr<float>(); const float* A = o_req.data_ptr<float>();
  at::parallel_for(0, R, 8, [&](int64_t b0, int64_t e0) {
    for (int64_t r = b0; r < e0; ++r) for (int64_t g = 0; g < hkv; ++g) for (int64_t i = 0; i < rep; ++i) {
      const float* a = A + ((g * R + r) * rep + i) * hd; float* d = O + r * hkv * rep * hd + (g * rep + i) * hd;
      #pragma omp simd
      for (int64_t j = 0; j < hd; ++j) d[j] += a[j];
    }
  });
}
"""


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline
        _EXT = load_inline(name='gct_edge_x86_kernels', cpp_sources=_SRC,
                           functions=['rms_norm', 'quant_rows', 'w8a8', 'silu_mul', 'qkv_prep', 'softmax2', 'combine', 'qkv_prep2', 'attn_own', 'add_req'],
                           extra_cflags=['-O3', '-march=native', '-fopenmp'], extra_ldflags=['-fopenmp'],
                           verbose=False)
    return _EXT


class W8A8:
    def __init__(self, weight):
        w = weight.detach().float()
        self.n, self.k = w.shape
        self.scale = (w.abs().amax(1).clamp_min(1e-20) / 127).contiguous()
        self.wqt = (w / self.scale[:, None]).round().clamp(-127, 127).to(torch.int8).t().contiguous()
        self.e = ext()

    def __call__(self, x, res=None):
        shape = x.shape
        flat = x.reshape(-1, self.k).float().contiguous()
        if flat.shape[0] == 0:
            return flat.new_zeros(*shape[:-1], self.n)
        r = None if res is None else res.reshape(-1, self.n).float().contiguous()
        return self.e.w8a8(flat, self.wqt, self.scale, r).reshape(*shape[:-1], self.n)


class Dense:
    def __init__(self, weight):
        self.wt = weight.detach().float().t().contiguous()
        self.n, self.k = weight.shape

    def __call__(self, x, res=None):
        flat = x.reshape(-1, self.k).float()
        out = flat @ self.wt if res is None else torch.addmm(res.reshape(-1, self.n).float(), flat, self.wt)
        return out.reshape(*x.shape[:-1], self.n)


def make(kind):
    return {'w8a8': W8A8, 'fp32': Dense}[kind]
