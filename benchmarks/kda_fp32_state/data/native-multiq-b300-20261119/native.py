"""Native Q1/multi-Q outputs and terminal state; no checkpoint snapshots in timing."""
import gc,hashlib,json,math,os,statistics,sys,time
from pathlib import Path
import torch
import cutlass
from vllm.models.kimi_k3.nvidia import kda_fp32_state as kda
from external.adapters import fixture,packed,NAMES,cf_runtime,fi_single

ROOT=Path(__file__).parent
RESULT=Path(os.environ['RESULT_DIR'])

def log(stage,**kw):print(json.dumps(dict(time=time.time(),stage=stage,**kw)),flush=True)
def bits(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    assert bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
    assert torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))

def equal(a,b,c):
    bits(a[0],b[0])
    if c:
        bits(a[2],b[2]);pos=a[2];assert bool(((pos>=0)&(pos<c)).all())
        mask=torch.arange(a[1].shape[2],device='cuda')[None,None,:]<pos[:,:,None]
        if c==16:mask[:,:,16]=pos>0
        bits(a[1][mask],b[1][mask])

class Engine:
    def __init__(self,name,x,anchor):
        self.name,self.x=name,x;self.b,self.q,self.h=x['q'].shape[:3]
        b,q,h=self.b,self.q,self.h
        self.chunk=16 if name=='cfbi_c16' else 64 if name=='cfbi_c64' else 0
        dtype=torch.bfloat16 if name=='flashinfer' else torch.float32
        self.back=[torch.cat([torch.zeros_like(anchor[:1]),anchor]).to(dtype)]
        if self.chunk:
            self.runtime=cf_runtime(self.chunk)
            self.back.extend([torch.zeros(b+1,h,self.runtime.ROWS,self.runtime.ROWF,device='cuda'),torch.zeros(b+1,h,device='cuda',dtype=torch.int32)])
        self.pool=[s[1:] for s in self.back]
        self.ids=torch.arange(b,device='cuda',dtype=torch.int32)
        self.cfids=self.ids+1
        self.output=torch.empty_like(x['q'])
        if self.chunk:
            data=packed(x);cu=torch.arange(b+1,device='cuda',dtype=torch.int32)*q
            if q==1:
                self.runtime._cu_ones(b,x['q'].device)
                self.kernel=lambda:self.runtime.decode(*data,x['A_log'],x['dt_bias'],self.cfids,*self.back,-5.,out=self.output.view(1,b,h,128))
            else:
                # rows_fast is the public general multi-row API. In v5 the
                # slots entry can dispatch a Q1-only dsync9 kernel and trap.
                dec=self.runtime.kernels()[1]
                self.kernel=lambda:self.output.copy_(dec.rows_fast(*data,x['A_log'],x['dt_bias'],cu,*self.pool,-5.).view_as(self.output))
                # Retain the returned output directly: no output copy in measured work.
                def run():self.output=dec.rows_fast(*data,x['A_log'],x['dt_bias'],cu,*self.pool,-5.).view_as(x['q'])
                self.kernel=run
        elif name=='fp32_state':
            self.plan=kda.prepare_decode(**x,state_in=self.pool[0],state_out=self.pool[0],out=self.output)
            self.kernel=self.plan.run
        else:
            assert q==1
            self.kernel=fi_single(x,self.pool[0],self.ids,self.output)
        self.initial=[s.clone() for s in self.pool]
    def run(self):self.kernel()
    def save(self):
        for a,b in zip(self.initial,self.pool):a.copy_(b)
    def reset(self):
        for a,b in zip(self.pool,self.initial):a.copy_(b)
    def step(self,x,rows=None):
        rows=self.ids if rows is None else rows
        b=rows.numel();out=torch.empty(b,1,self.h,128,device='cuda',dtype=torch.bfloat16)
        if self.chunk:self.runtime.decode(*packed(x),x['A_log'],x['dt_bias'],rows+1,*self.back,-5.,out=out.view(1,b,self.h,128))
        elif self.name=='fp32_state':kda.prepare_decode(**x,state_in=self.pool[0],state_out=self.pool[0],out=out,state_indices=rows).run()
        else:fi_single(x,self.pool[0],rows,out)()
        return out

def history(engines,x,lengths):
    for t in range(max(lengths,default=0)):
        ix=torch.tensor([i for i,l in enumerate(lengths) if l>t],device='cuda',dtype=torch.long)
        part={k:x[k][ix,t:t+1].contiguous() for k in NAMES};part.update(A_log=x['A_log'],dt_bias=x['dt_bias'])
        for e in engines:e.step(part,ix.int())
    for e in engines:e.save()

def create(q,h,seed,context):
    b=len(context);x=fixture(b,q,h,seed)
    gen=torch.Generator(device='cuda').manual_seed(seed+1)
    anchor=(torch.randn(b,h,128,128,device='cuda',generator=gen)*.2).bfloat16().float()
    names=['fp32_state','cfbi_c16','cfbi_c64']+(['flashinfer'] if q==1 else [])
    engines=[Engine(n,x,anchor) for n in names]
    hist=fixture(b,64,h,seed+2);hist.update(A_log=x['A_log'],dt_bias=x['dt_bias'])
    history(engines,hist,[v%64 for v in context]);return engines,hist

def check(e):
    e.reset();e.run();out=e.output.clone();state=[s.clone() for s in e.pool]
    e.reset()
    for t in range(e.q):
        part={k:e.x[k][:,t:t+1].contiguous() for k in NAMES};part.update(A_log=e.x['A_log'],dt_bias=e.x['dt_bias'])
        bits(e.step(part)[:,0],out[:,t])
    equal(e.pool,state,e.chunk)
    # Repeat and graph must retain the same complete effective state.
    e.reset();e.run();e.reset()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):e.run()
    graphout=e.output
    e.reset();graph.replay();bits(graphout,out);equal(e.pool,state,e.chunk)
    e.reset()

def capture(e,cycles=1):
    e.reset();e.run();e.reset()
    graph=torch.cuda.CUDAGraph();a=torch.cuda.Event(enable_timing=True,external=True);z=torch.cuda.Event(enable_timing=True,external=True)
    with torch.cuda.graph(graph):
        a.record()
        for _ in range(cycles):e.run()
        z.record()
    # Keep output storage alive for each captured allocation.
    return graph,a,z,e.output

def timing(engines,groups=10,warmup=2,cycles=1,slots=8):
    graphs={e.name:[capture(e,cycles) for _ in range(slots)] for e in engines}
    samples={e.name:[] for e in engines}
    for group in range(groups+warmup):
        pending=[]
        for slot in range(slots):
            shift=(group+slot)%len(engines)
            for e in engines[shift:]+engines[:shift]:
                e.reset();g,a,z,_=graphs[e.name][slot];g.replay();pending.append((e.name,a,z))
        torch.cuda.synchronize()
        if group>=warmup:
            for name,a,z in pending:samples[name].append(a.elapsed_time(z)*1000/cycles)
    med={n:statistics.median(s) for n,s in samples.items()}
    return dict(us=med,samples_us=samples,relative_to_fp32={n:v/med['fp32_state'] for n,v in med.items()},calls_per_graph=cycles)

def save(name,rows):(RESULT/(name+'.json')).write_text(json.dumps(rows,indent=2))

def main_cases(smoke):
    cases=json.loads((ROOT/'workloads.json').read_text())['cases'];rows=[]
    if smoke:cases=[c for c in cases if c['nominal_kv']==8192]
    for case in cases:
        for h in ([12] if smoke else [12,16,32]):
            log('prepare',case=case['case_id'],h=h)
            seed=8841+int(hashlib.sha256(case['case_id'].encode()).hexdigest()[:6],16)
            engines,_=create(case['q'],h,seed,case['context'])
            for e in engines:check(e)
            row=dict(case=case['case_id'],q=case['q'],b=64,h=h,seed=seed,context=case['context'],**timing(engines,groups=2 if smoke else 10))
            rows.append(row);save('smoke' if smoke else 'main',rows);log('measured',done=len(rows),case=case['case_id'],h=h,us=row['us'])
            del engines,e;gc.collect()

def phases():
    rows=[];cycles=[]
    for q in [1,4,6,8]:
        for h in [12,16,32]:
            engines,hist=create(q,h,1900+q+h,[0]*64)
            for pos in range(64):
                for e in engines:check(e)
                row=dict(q=q,h=h,b=64,pos=pos,**timing(engines,groups=3,warmup=1,slots=4));rows.append(row);save('positions',rows)
                for e in engines:
                    e.reset();part={k:hist[k][:,pos:pos+1].contiguous() for k in NAMES};part.update(A_log=hist['A_log'],dt_bias=hist['dt_bias']);e.step(part);e.save()
                log('position',q=q,h=h,pos=pos,done=len(rows),total=768)
            calls=64//math.gcd(q,64)
            row=dict(q=q,h=h,b=64,period_calls=calls,period_tokens=calls*q,**timing(engines,groups=5,warmup=1,cycles=calls,slots=2));cycles.append(row);save('cycles',cycles);log('cycle',q=q,h=h,us=row['us'])
            del engines,e;gc.collect()

@torch.no_grad()
def main():
    assert sys.flags.no_user_site and torch.cuda.device_count()==2
    assert all(torch.cuda.get_device_name(i)=='NVIDIA B300 SXM6 AC' for i in range(2))
    save('environment',dict(torch=torch.__version__,cuda=torch.version.cuda,dsl=cutlass.__version__,python=sys.version,gpus=[torch.cuda.get_device_name(i) for i in range(2)],snapshot=os.environ['SOURCE_DIGEST']))
    mode=sys.argv[1]
    if mode in ('smoke','main'):main_cases(mode=='smoke')
    elif mode=='phases':phases()
    log('COMPLETE',mode=mode)
if __name__=='__main__':main()
