"""Q4/Q6 paired complete APIs, upstream CF-BI position/cycle coverage."""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import torch
import cutlass
from adapters import Engine,fixture,advance_history,NAMES
from tests.helpers import bits
import kda_consistent

ROOT=Path(__file__).parent
RESULT=Path(os.environ['RESULT_DIR']);RESULT.mkdir(parents=True,exist_ok=True)
NAMES_IMPL=['fp32_state','flashinfer','cfbi_c16','cfbi_c64']


def log(stage,**kw):print(json.dumps(dict(time=time.time(),stage=stage,**kw)),flush=True)


def equal_state(a,b,chunk):
    bits(a[0],b[0])
    if not chunk:return
    bits(a[2],b[2]);pos=a[2]
    assert bool(((pos>=0)&(pos<chunk)).all())
    mask=torch.arange(a[1].shape[2],device='cuda')[None,None,:]<pos[:,:,None]
    if chunk==16:mask[:,:,16]=pos>0
    bits(a[1][mask],b[1][mask])


def check(e,graph=None):
    e.reset();e.run()
    out=e.output.clone();states=[s.clone() for s in e.pool];cps=[s.clone() for s in e.cp]
    ids=e.ids.long();counts=e.counts.long();req=torch.arange(e.b,device='cuda')
    for i,s in enumerate(states):
        if e.chunk:want=cps[i].flatten(0,1).index_select(0,counts*e.b+req)
        else:
            want=e.initial[i].index_select(0,ids).clone()
            active=counts>0
            want[active]=cps[i][req[active],counts[active]-1]
        equal_state([s.index_select(0,ids)],[want],0)
        mask=torch.ones(s.shape[0],device='cuda',dtype=torch.bool);mask[ids]=False
        bits(s[mask],e.initial[i][mask])
    if e.name!='flashinfer':
        # Independent direct Q1 chain checks every effective checkpoint. The FI
        # grouped path has a different rounding contract from repeated BF16 Q1.
        e.reset()
        for t in range(e.q):
            part={k:e.x[k][:,t:t+1].contiguous() for k in NAMES}
            part.update(A_log=e.x['A_log'],dt_bias=e.x['dt_bias'])
            yy=e.step(part)
            bits(yy[:,0],out[:,t])
            want=[s[t+1] for s in cps] if e.chunk else [cps[0][:,t]]
            equal_state(e.selected(),want,e.chunk)
    else:
        for s in cps:assert bool(torch.isfinite(s).all())
        compact=e.initial[0].index_select(0,ids)
        ref=Engine('flashinfer',e.x,compact,torch.arange(e.b,device='cuda',dtype=torch.int32),e.counts)
        ref.run();bits(ref.output,out);bits(ref.cp[0],cps[0])
        bits(ref.pool[0],states[0].index_select(0,ids))
    if graph is not None:
        e.reset();graph.replay();bits(e.output,out)
        for got,want in zip(e.pool,states):bits(got,want)
        for got,want in zip(e.cp,cps):bits(got,want)
    e.reset()


def capture(e,cycles=1):
    e.reset();e.run();e.reset()
    graph=torch.cuda.CUDAGraph()
    start=torch.cuda.Event(enable_timing=True,external=True)
    end=torch.cuda.Event(enable_timing=True,external=True)
    with torch.cuda.graph(graph):
        start.record()
        for _ in range(cycles):e.run()
        end.record()
    return graph,start,end


def timing(engines,groups=10,warmup=2,cycles=1,slots=8):
    graphs={e.name:[capture(e,cycles) for _ in range(slots)] for e in engines}
    samples={e.name:[] for e in engines}
    for group in range(groups+warmup):
        pending=[]
        for slot in range(slots):
            shift=(group+slot)%len(engines)
            for e in engines[shift:]+engines[:shift]:
                e.reset();g,a,z=graphs[e.name][slot];g.replay();pending.append((e.name,a,z))
        torch.cuda.synchronize()
        if group>=warmup:
            for name,a,z in pending:samples[name].append(a.elapsed_time(z)*1000/cycles)
    med={n:statistics.median(s) for n,s in samples.items()}
    return dict(us=med,samples_us=samples,ours_over={n:med['fp32_state']/v for n,v in med.items() if n!='fp32_state'},calls_per_graph=cycles)


def create(b,q,h,seed,context):
    x=fixture(b,q,h,seed)
    gen=torch.Generator(device='cuda').manual_seed(seed+1)
    anchor=(torch.randn(2*b+1,h,128,128,device='cuda',generator=gen)*.2).bfloat16()
    ids=torch.arange(b,device='cuda',dtype=torch.int32).flip(0)*2+1
    counts=torch.arange(b,device='cuda',dtype=torch.int32)%(q+1)
    engines=[Engine(n,x,anchor,ids,counts) for n in NAMES_IMPL]
    history=fixture(b,64,h,seed+2);history.update(A_log=x['A_log'],dt_bias=x['dt_bias'])
    advance_history(engines,history,[n%64 for n in context])
    return engines,history


def save(name,rows):
    (RESULT/(name+'.json')).write_text(json.dumps(rows,indent=2))


def main_cases(smoke):
    cases=json.loads((ROOT/'workloads.json').read_text())['cases']
    if smoke:cases=cases[:2]
    rows=[]
    for case in cases:
        for h in ([12] if smoke else [12,16,32]):
            seed=8841+int(hashlib.sha256(case['case_id'].encode()).hexdigest()[:6],16)
            log('prepare_start',case=case['case_id'],h=h)
            engines,_=create(64,case['q'],h,seed,case['context'])
            for e in engines:
                log('check_start',implementation=e.name)
                probe=capture(e);g=probe[0];check(e,g)
                # Graph metadata remains live: acceptance and slots are device values.
                if smoke:
                    old_ids=e.ids.clone();old_counts=e.counts.clone()
                    e.counts.zero_();check(e,g)
                    e.counts.fill_(e.q);e.ids.copy_(old_ids.flip(0));check(e,g)
                    e.ids.copy_(old_ids);e.counts.copy_(old_counts)
            log('correctness_pass',case=case['case_id'],h=h)
            row=dict(case=case['case_id'],b=64,q=case['q'],h=h,seed=seed,context=case['context'],history_suffix=[n%64 for n in case['context']],**timing(engines,groups=2 if smoke else 10))
            rows.append(row);save('smoke' if smoke else 'main',rows)
            log('measured',case=row['case'],h=h,us=row['us'],done=len(rows),total=len(cases)*(1 if smoke else 3))
            del engines,e,g;gc.collect()
    return rows


def phases():
    rows=[];cycles=[]
    for q in [4,6]:
        for h in [12,16,32]:
            engines,history=create(64,q,h,1900+q+h,[0]*64)
            for pos in range(64):
                for e in engines:
                    probe=capture(e);g=probe[0];check(e,g)
                row=dict(b=64,q=q,h=h,pos=pos,**timing(engines,groups=3,warmup=1,slots=4))
                rows.append(row);save('positions',rows)
                for e in engines:
                    e.reset()
                    part={k:history[k][:,pos:pos+1].contiguous() for k in NAMES}
                    part.update(A_log=history['A_log'],dt_bias=history['dt_bias'])
                    e.step(part);e.save()
                log('position',q=q,h=h,pos=pos,us=row['us'],done=len(rows),total=384)
            # A full accepted Q at each call advances through the entire phase
            # period: Q4=16 calls/64 tokens; Q6=32 calls/192 tokens.
            for e in engines:e.counts.fill_(q)
            calls=64//math.gcd(q,64)
            row=dict(b=64,q=q,h=h,period_calls=calls,period_tokens=calls*q,**timing(engines,groups=5,warmup=1,slots=2,cycles=calls))
            cycles.append(row);save('cycles',cycles)
            log('cycle',q=q,h=h,us=row['us'])
            del engines,e,g;gc.collect()



def accuracy():
    from tests.fp64_reference import run as fp64
    rows=[]
    for q in [4,6]:
        for gate in ['normal','open','saturated']:
            b,h=2,3
            full=fixture(b,64,h,20261119+q)
            if gate!='normal':
                full['A_log'].zero_();full['dt_bias'].zero_();full['g'].fill_(-1000)
                if gate=='saturated':
                    full['g'][...,::2]=1000;full['beta'][...,::2]=1000;full['beta'][...,1::2]=-1000
            x={k:full[k][:,64-q:].contiguous() for k in NAMES}
            x.update(A_log=full['A_log'],dt_bias=full['dt_bias'])
            gen=torch.Generator(device='cuda').manual_seed(20261119)
            anchor=(torch.randn(2*b+1,h,128,128,device='cuda',generator=gen)*.2).bfloat16()
            ids=torch.arange(b,device='cuda',dtype=torch.int32)*2+1
            counts=torch.full((b,),q,device='cuda',dtype=torch.int32)
            engines=[Engine(n,x,anchor,ids,counts) for n in NAMES_IMPL]
            advance_history(engines,full,[64-q]*b)
            data=[full[k].view(1,b*64,*full[k].shape[2:]) for k in NAMES]
            ref=fp64(*data,full['A_log'],full['dt_bias'],anchor[ids.long()].float(),torch.arange(b+1,device='cuda',dtype=torch.int32)*64,full_precision=True)
            reference_out=ref['output'].view(b,64,h,128)[:,64-q:]
            for e in engines:
                e.run()
                if e.chunk:assert bool((e.selected()[2]==0).all())
                r=dict(q=q,gate=gate,implementation=e.name,history_tokens=64-q,total_tokens=64)
                for name,got,want in [('output',e.output,reference_out),('state',e.selected()[0],ref['state_out'])]:
                    assert bool(torch.isfinite(got).all()) and bool(torch.isfinite(want).all())
                    delta=got.double()-want
                    r[name]=dict(nrmse=float(delta.square().mean().sqrt()/want.square().mean().sqrt().clamp_min(1e-300)),max_normalized=float(delta.abs().max()/want.abs().max().clamp_min(1e-300)))
                rows.append(r)
            save('accuracy',rows);log('accuracy',q=q,gate=gate,done=len(rows),total=24)


@torch.no_grad()
def main():
    assert sys.flags.no_user_site and torch.cuda.device_count()==2
    assert all(torch.cuda.get_device_name(i)=='NVIDIA B300 SXM6 AC' for i in range(2))
    (RESULT/'environment.json').write_text(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,dsl=cutlass.__version__,python=sys.version,gpus=[torch.cuda.get_device_name(i) for i in range(2)],package=kda_consistent.__file__,snapshot=os.environ['SOURCE_DIGEST']),indent=2))
    mode=sys.argv[1]
    if mode in ('smoke','main'):main_cases(mode=='smoke')
    if mode=='smoke':accuracy()
    elif mode=='phases':phases()
    log('COMPLETE',mode=mode)

if __name__=='__main__':main()
