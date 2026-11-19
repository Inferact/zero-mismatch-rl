"""Prepared complete decode APIs, including all checkpoints and accepted state."""
import importlib.util
import os
from pathlib import Path
import torch
import cuda.bindings.driver as cuda
import fi_q1
import fi_q8
import kda_consistent as kda
from kda_consistent._checkpoint import compiled_commit
from upstream_adapter import modules

NAMES = ('q','k','v','g','beta')
_CF = {}


def cf_runtime(chunk):
    if chunk not in _CF:
        path=Path(__file__).parent/'upstream/vllm/models/kimi_k3/nvidia/k31_kda_cfbi.py'
        spec=importlib.util.spec_from_file_location(f'upstream_cfbi_{chunk}',path)
        m=importlib.util.module_from_spec(spec)
        previous=os.environ.get('K31_KDA_CFBI_CONTRACT')
        os.environ['K31_KDA_CFBI_CONTRACT']='v6' if chunk==16 else 'v5'
        try:spec.loader.exec_module(m)
        finally:
            if previous is None:os.environ.pop('K31_KDA_CFBI_CONTRACT',None)
            else:os.environ['K31_KDA_CFBI_CONTRACT']=previous
        ext=modules()
        m._EXT=(ext['v6'],ext['v6']) if chunk==16 else (ext['v5_pf'],ext['v5_dec'])
        _CF[chunk]=m
    return _CF[chunk]


def fixture(b,q,h,seed):
    gen=torch.Generator(device='cuda').manual_seed(seed)
    xs=[(torch.randn((b,q,h,128),device='cuda',generator=gen)*.2).bfloat16() for _ in range(4)]
    xs.append((torch.randn((b,q,h),device='cuda',generator=gen)*.2).bfloat16())
    return dict(zip(NAMES,xs),A_log=torch.randn(h,device='cuda',generator=gen)*.2,
                dt_bias=torch.randn(h,128,device='cuda',generator=gen)*.2)


def packed(x):return [x[k].view(1,-1,*x[k].shape[2:]) for k in NAMES]


def fi_single(x,pool,ids,out):
    b,h=ids.numel(),pool.shape[1]
    tile,red=fi_q1._select_kernel_schedule(128,1,True,b*h)
    fn=fi_q1._get_compiled_kernel(128,1,1,1,1,1,1,1,1,tile,red,1,0)
    cu=torch.arange(b+1,device='cuda',dtype=torch.int32)
    dummy=torch.zeros(1,device='cuda',dtype=torch.int32)
    args=(*packed(x),pool,pool,ids,out.view(1,b,h,128),x['A_log'],x['dt_bias'].flatten(),cu,ids,dummy,128**-.5,1e-6,-5.)
    return lambda:fn(*args)


class Engine:
    def __init__(self,name,x,anchor,ids,counts):
        self.name,self.x,self.ids,self.counts=name,x,ids,counts
        self.b,self.q,self.h=x['q'].shape[:3]
        self.chunk=16 if name=='cfbi_c16' else 64 if name=='cfbi_c64' else 0
        b,q,h=self.b,self.q,self.h
        self.output=torch.empty_like(x['q'])
        self.idlong=ids.long()
        if self.chunk:
            self.runtime=cf_runtime(self.chunk)
            self.pool=[anchor.float().clone(),torch.zeros(anchor.shape[0],h,self.runtime.ROWS,self.runtime.ROWF,device='cuda'),torch.zeros(anchor.shape[0],h,device='cuda',dtype=torch.int32)]
            self.cp=[torch.empty(q+1,b,*s.shape[1:],device='cuda',dtype=s.dtype) for s in self.pool]
            self.row_out=torch.empty(q,1,b,h,128,device='cuda',dtype=torch.bfloat16)
            self.row_inputs=[[x[k][:,t].unsqueeze(0) for k in NAMES] for t in range(q)]
            self.request_ids=torch.arange(b,device='cuda',dtype=torch.int64)
            # Initialize the upstream metadata cache before capture.
            self.runtime._cu_ones(b,x['q'].device)
        else:
            dtype=torch.float32 if name=='fp32_state' else torch.bfloat16
            self.pool=[anchor.to(dtype).clone()]
            self.cp=[torch.empty(b,q,h,128,128,device='cuda',dtype=dtype)]
            if name=='fp32_state':
                self.plan=kda.prepare_decode(**x,state_in=self.pool[0],state_out=self.pool[0],out=self.output,
                    checkpoint_states=self.cp[0],state_indices=ids,num_accepted_tokens=counts,state_dtype=dtype)
                self.kernel=self.plan.run;self.commit=self.plan.commit
            else:
                cu=torch.arange(b+1,device='cuda',dtype=torch.int32)*q
                cpids=torch.arange(b*q,device='cuda',dtype=torch.int32).view(b,q)
                dummy=torch.zeros(b,device='cuda',dtype=torch.int32)
                args=(*packed(x),x['A_log'],x['dt_bias'].flatten(),cu,cpids,dummy,self.cp[0].view(b*q,h,128,128),self.pool[0],ids,self.output.view(1,b*q,h,128))
                fn=fi_q8._get_grouped_compiled((128,q,h,h,2,1,1,1,1),*args)
                self.kernel=lambda:fn(*args,b,b*q,h*128,h*128*128,h*128*128,1,128**-.5,-5.,cuda.CUstream(torch.cuda.current_stream().cuda_stream))
                c=compiled_commit(torch.cuda.current_device(),torch.cuda.get_device_capability(),False,True,True)
                self.commit=lambda:c(self.pool[0],self.cp[0],self.pool[0],counts,ids)
        self.initial=[p.clone() for p in self.pool]

    def save(self):
        for a,b in zip(self.initial,self.pool):a.copy_(b)

    def reset(self):
        for a,b in zip(self.pool,self.initial):a.copy_(b)

    def step(self,x,ids=None):
        ids=self.ids if ids is None else ids
        b=ids.numel();out=torch.empty(b,1,self.h,128,device='cuda',dtype=torch.bfloat16)
        if self.chunk:
            self.runtime.decode(*packed(x),x['A_log'],x['dt_bias'],ids,*self.pool,-5.,out=out.view(1,b,self.h,128))
        elif self.name=='fp32_state':
            kda.prepare_decode(**x,state_in=self.pool[0],state_out=self.pool[0],out=out,state_indices=ids,state_dtype=torch.float32).run()
        else:fi_single(x,self.pool[0],ids,out)()
        return out

    def run(self):
        if not self.chunk:
            self.kernel();self.commit();return
        self.idlong=self.ids.long()
        for cp,s in zip(self.cp,self.pool):torch.index_select(s,0,self.idlong,out=cp[0])
        for t in range(self.q):
            self.runtime.decode(*self.row_inputs[t],self.x['A_log'],self.x['dt_bias'],self.ids,*self.pool,-5.,out=self.row_out[t])
            self.output[:,t].copy_(self.row_out[t,0])
            for cp,s in zip(self.cp,self.pool):torch.index_select(s,0,self.idlong,out=cp[t+1])
        ix=self.counts.long()*self.b+self.request_ids
        for cp,s in zip(self.cp,self.pool):s.index_copy_(0,self.idlong,cp.flatten(0,1).index_select(0,ix))

    def selected(self):return [s.index_select(0,self.ids.long()) for s in self.pool]

    def checkpoint(self,t):
        if self.chunk:return [s[t+1] for s in self.cp]
        return [self.cp[0][:,t]]


def advance_history(engines,x,lengths):
    # The supplied anchor is at a closed 64-token boundary; compute the actual
    # remaining suffix rather than synthesizing pos or row-cache contents.
    for t in range(max(lengths,default=0)):
        rows=[i for i,n in enumerate(lengths) if n>t]
        index=torch.tensor(rows,device='cuda',dtype=torch.long)
        part={k:x[k][index,t:t+1].contiguous() for k in NAMES}
        part.update(A_log=x['A_log'],dt_bias=x['dt_bias'])
        for e in engines:e.step(part,e.ids.index_select(0,index))
    for e in engines:e.save()
