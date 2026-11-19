"""Thin adapter for unchanged public CUDA extensions; no framework dependency."""
from pathlib import Path
import os
import torch
from torch.utils.cpp_extension import load

_modules = None


def modules():
    global _modules
    if _modules is None:
        root = Path(__file__).resolve().parent / 'upstream/csrc/k31/kda'
        _modules = {}
        for name, filename in [('v6', 'kda_cfbi_v6.cu'), ('v5_pf', 'kda_cfbi6_pf4.cu'),
                               ('v5_dec', 'kda_cfbi6_dsync8.cu')]:
            directory = Path(os.environ['TORCH_EXTENSIONS_DIR']) / name
            directory.mkdir(parents=True, exist_ok=True)
            _modules[name] = load(name='public_cfbi_' + name, sources=[str(root / filename)],
                build_directory=str(directory), extra_cflags=['-O3', '-UNDEBUG'],
                extra_cuda_cflags=['-O3', '--fmad=false', '-UNDEBUG', '-lineinfo',
                                  '-gencode=arch=compute_103a,code=sm_103a'], verbose=True)
    return _modules


class PublicCF:
    def __init__(self, initial, chunk):
        self.c = chunk
        self.b, self.h = initial.shape[:2]
        self.dev = initial.device
        self.E = initial.float().clone()
        self.C = torch.full((self.b, self.h, 17 if chunk == 16 else 64,
                             256 if chunk == 16 else 640), float('nan'), device=self.dev)
        self.P = torch.zeros((self.b, self.h), device=self.dev, dtype=torch.int32)
        ms = modules()
        self.pf = ms['v6' if chunk == 16 else 'v5_pf']
        self.dec = ms['v6' if chunk == 16 else 'v5_dec']

    @property
    def state(self):
        return [self.E, self.C, self.P]

    def reset(self, initial):
        self.E.copy_(initial)
        self.P.zero_()

    def bind_row(self, xs, a, bias, *, slots=False):
        length = xs[0].shape[1]
        data = [x.view(1, -1, *x.shape[2:]) for x in xs]
        cu = torch.arange(self.b + 1, device=self.dev, dtype=torch.int32) * length
        box = {}
        if slots:
            # Slot zero is padding upstream. Allocate a real pool with slot zero reserved.
            pool = [torch.cat([torch.zeros_like(x[:1]), x], dim=0) for x in self.state]
            self.E, self.C, self.P = [x[1:] for x in pool]
            ids = torch.arange(1, self.b + 1, device=self.dev, dtype=torch.int32)
            out = torch.empty_like(data[2])
            def run():
                box['out'] = self.dec.rows_fast_slots(*data, a, bias, cu, *pool, ids, -5., out)
        else:
            def run():
                box['out'] = self.dec.rows_fast(*data, a, bias, cu, *self.state, -5.)
        return run, box

    def bind_decode(self, steps, a, bias):
        # Build the pool once, so every captured token reads and writes the same state.
        pool = [torch.cat([torch.zeros_like(x[:1]), x], dim=0) for x in self.state]
        self.E, self.C, self.P = [x[1:] for x in pool]
        cu = torch.arange(self.b + 1, device=self.dev, dtype=torch.int32)
        ids = torch.arange(1, self.b + 1, device=self.dev, dtype=torch.int32)
        calls, outputs = [], []
        for xs in steps:
            data = [x.view(1, -1, *x.shape[2:]) for x in xs]
            out = torch.empty_like(data[2])
            def run(data=data, out=out):
                self.dec.rows_fast_slots(*data, a, bias, cu, *pool, ids, -5., out)
            calls.append(run)
            outputs.append(out.view_as(xs[0]))
        return calls, outputs

    def bind_prefill(self, xs, a, bias):
        length = xs[0].shape[1]
        data = [x.view(1, -1, *x.shape[2:]) for x in xs]
        n = (length + self.c - 1) // self.c
        tab = [[b, b * length + j * self.c, min(self.c, length - j * self.c), int(j == n - 1)]
               for b in range(self.b) for j in range(n)]
        table = torch.tensor(tab, device=self.dev, dtype=torch.int32).reshape(-1, 4)
        offsets = torch.arange(self.b + 1, device=self.dev, dtype=torch.int32) * n
        bv = 316 if self.b == 1 else 332
        box = {}
        def run():
            box['out'] = self.pf.prefill(*data, a, bias, table, offsets, *self.state, -5., bv)[0]
        return run, box

    def transfer(self, other):
        self.E.copy_(other.E)
        self.P.copy_(other.P)
        for b in range(self.b):
            for h in range(self.h):
                pos = int(other.P[b, h])
                self.C[b, h, :pos].copy_(other.C[b, h, :pos])
                if self.c == 16 and pos:
                    self.C[b, h, 16].copy_(other.C[b, h, 16])

    def compare(self, other):
        from common import bits
        bits(self.E, other.E)
        bits(self.P, other.P)
        for b in range(self.b):
            for h in range(self.h):
                pos = int(self.P[b, h])
                assert 0 <= pos < self.c
                bits(self.C[b, h, :pos], other.C[b, h, :pos])
                if self.c == 16 and pos:
                    bits(self.C[b, h, 16], other.C[b, h, 16])

    def workspace_bytes(self, length):
        chunks = self.b * ((length + self.c - 1) // self.c) * self.h
        return chunks * (4 * (4 * 16 * 128 + 2 * 16 * 16 + 2 * 128) if self.c == 16
                         else 4 * (16384 + 64 * 128 + 64 * 64 + 8192 + 128))
