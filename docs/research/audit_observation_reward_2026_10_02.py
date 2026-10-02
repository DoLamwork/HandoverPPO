# Read-only reproducibility script. Run from repository root with the project venv.
# Outputs diagnostics to stdout; does not modify data, checkpoints, or source code.
import sys, json, copy, collections
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, 'src')
from ho_optim_drl.config import Config
from ho_optim_drl.data import load_radio_trace, load_cell_topology
from ho_optim_drl.protocol import HandoverEngine
from ho_optim_drl.smdp import TopologyCandidateObservationBuilder, HandoverRewardModel
from ho_optim_drl.agents.ppo.trainer import SMDPPPO
torch.set_num_threads(1)
root=Path('results/models/uav_100m_grid100m_topology109_v1_s17')
manifest=json.loads((root/'dataset_split_manifest.json').read_text())
agent, extra=SMDPPPO.from_checkpoint(root/'smdp_ppo_final.pt',device='cpu')
config=Config(**extra['protocol_config'])
print('MODEL',json.dumps({'steps':agent.total_macro_steps,'config':vars(agent.config)}),flush=True)
data='data/uav_100m_grid100m_dev_processed'
topology=load_cell_topology(data)
rng=np.random.default_rng(17)
chosen=[]
for speed in (30,50,70,90):
    group=[r for r in manifest['splits']['train'] if r['speed_kmh']==speed]
    chosen.extend([group[int(j)] for j in rng.choice(len(group),6,replace=False)])
stats={}
counter=[]
reward_model=HandoverRewardModel(config)
for mode in ('3gpp','ppo'):
    rows=[]; probs_all=[]; durations={0:[],1:[]}; actions=collections.Counter()
    all_masks=[]; traces_done=[]
    for rec in chosen:
        trace=load_radio_trace(config,data,rec['rsrp_file'],rec['sinr_file'])
        engine=HandoverEngine(config)
        builder=TopologyCandidateObservationBuilder(config,topology=topology)
        engine.reset(trace);builder.reset()
        prior_t=None;prior_action=None; decision_idx=0; cf_per_trace=0
        while True:
            tick=engine.advance_one_tick();builder.observe(tick)
            if tick.terminated:
                if prior_t is not None: durations[int(prior_action!=0)].append((engine.snapshot.tic-prior_t)*.01)
                break
            if engine.decision_point is None: continue
            snap=engine.snapshot; obs=builder.build(snap); cells=np.array(builder.candidate_cells,dtype=int)
            mask=builder.action_mask();all_masks.append(int(mask.sum()))
            with torch.no_grad():
                dist,_=agent.model.distribution_and_value(torch.tensor(obs).unsqueeze(0),torch.tensor(mask).unsqueeze(0))
                probs=dist.probs[0].numpy()
            probs_all.append(probs)
            act=1 if mode=='3gpp' else int(np.argmax(probs))
            if prior_t is not None: durations[int(prior_action!=0)].append((snap.tic-prior_t)*.01)
            prior_t=snap.tic;prior_action=act;actions[act]+=1
            event_records=[r for r in builder._all_cell_history if snap.decision_point.a3_start_tic <= r.tic <= snap.tic and r.serving_cell==snap.serving_cell]
            sinr=snap.sinr_rate_db[cells]; future_t=snap.tic+9
            future=trace.sinr_rlm_db[cells,future_t] if future_t < trace.num_samples else np.full(10,np.nan)
            eff=np.log2(1+10**(sinr/10))
            rows.append({'history_ms':(event_records[-1].tic-event_records[0].tic)*10,'valid_alt_now':int(np.sum(sinr[1:]>-6)), 'c1_suitable':int(sinr[0]>-6),
                         'unsafe_alt_frac':float(np.mean(sinr[1:]<=-6)), 'alt_better_now':int(np.max(eff[1:])>eff[0]),
                         'c1_future_safe':float(future[0]>-6) if np.isfinite(future[0]) else np.nan,'alt_future_safe_frac':float(np.mean(future[1:]>-6)) if np.isfinite(future[0]) else np.nan,
                         'c1_above10':int(sinr[0]>10),'serving_above10':int(snap.sinr_rate_db[snap.serving_cell]>10),
                         'n310':snap.n310_progress,'t310':int(snap.t310_active)})
            if mode=='3gpp' and decision_idx % 10 == 0 and cf_per_trace<4 and snap.tic+100<trace.num_samples:
                outcomes=[]
                for a in range(11):
                    branch=copy.deepcopy(engine,{id(trace):trace})
                    if a==0:branch.stay()
                    else:branch.start_handover(int(cells[a-1]))
                    rate_sum=0.;ret=0.;hf=rf=pp=0
                    for j in range(100):
                        if branch.decision_point is not None:branch.start_handover()
                        z=branch.advance_one_tick()
                        rate_sum+=z.delivered_rate_mbps*.01
                        ret+=reward_model.compute(z).total
                        hf+=int(z.handover_failed);rf+=int(z.rlf_started);pp+=int(z.ping_pong_detected)
                    outcomes.append([rate_sum,ret,hf,rf,pp])
                counter.append({'trace':rec['trace_id'],'tic':snap.tic,'outcomes':outcomes})
                cf_per_trace+=1
            decision_idx+=1
            if act==0:engine.stay()
            else:engine.start_handover(int(cells[act-1]))
        traces_done.append(rec['trace_id'])
        if len(traces_done)%6==0:print('PROGRESS',mode,len(traces_done),flush=True)
    p=np.array(probs_all)
    stats[mode]={'decisions':len(rows),'actions':dict(actions),'all_masks_11':bool(all(v==11 for v in all_masks)),
                 'mean_probabilities':p.mean(axis=0).tolist(),
                 'prob_alt_percentiles':np.percentile(p[:,2:].sum(axis=1),[0,50,90,99,100]).tolist(),
                 'means':{k:float(np.nanmean([r[k] for r in rows])) for k in rows[0]},
                 'history_ms_values':sorted(set(r['history_ms'] for r in rows)),
                 'duration_seconds':{str(k):{'n':len(v),'percentiles_50_90_95_99_max':np.percentile(v,[50,90,95,99,100]).tolist()} for k,v in durations.items() if v}}
cf=np.array([r['outcomes'] for r in counter])
better_rate=cf[:,2:,0]>cf[:,1:2,0]
no_more_fail=(cf[:,2:,2]<=cf[:,1:2,2])&(cf[:,2:,3]<=cf[:,1:2,3])&(cf[:,2:,4]<=cf[:,1:2,4])
summary={'routes':len(chosen),'route_ids':[r['trace_id'] for r in chosen],'stats':stats,
         'counterfactual':{'states':len(counter),'horizon_s':1,'continuation':'always HO_C1 at subsequent decision','coverage':'up to 4 states per route at decisions 0,10,20,30; train only',
                           'any_alt_rate_better_frac':float(better_rate.any(axis=1).mean()),
                           'any_alt_rate_better_no_more_hof_rlf_pp_frac':float((better_rate&no_more_fail).any(axis=1).mean()),
                           'any_alt_old_reward_better_frac':float((cf[:,2:,1]>cf[:,1:2,1]).any(axis=1).mean()),
                           'best_rate_action_counts':dict(collections.Counter(map(int,cf[:,:,0].argmax(axis=1)))),
                           'best_old_reward_action_counts':dict(collections.Counter(map(int,cf[:,:,1].argmax(axis=1))))}}
print('AUDIT_RESULT',json.dumps(summary,ensure_ascii=False),flush=True)
