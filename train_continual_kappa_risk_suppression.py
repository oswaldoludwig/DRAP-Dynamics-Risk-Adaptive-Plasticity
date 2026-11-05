#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data_utils import SuiteSpec, load_suite_dataset
from kappatune_vla import TensorGroupPatterns, apply_selection, select_low_kappa_tensors, summarize_selection
from risk_regularization import RiskAugmentedDataset
from risk_weighted_tensor_importance import estimate, load_state, merge, save
from dynamics_risk_adjustment_suppression import DynamicsRiskSuppressedAdamW
from train_continual import load_model_and_processor, compute_action_bounds, make_collate_fn

REPOS={"libero_spatial":"lerobot/libero_spatial_image","libero_object":"lerobot/libero_object_image",
       "libero_goal":"lerobot/libero_goal_image","libero_long":"lerobot/libero_10_image",
       "libero_90":"lerobot/libero_90_image"}

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--suite",required=True,choices=list(REPOS))
    p.add_argument("--init-checkpoint",required=True); p.add_argument("--out-dir",required=True)
    p.add_argument("--risk-scores",required=True); p.add_argument("--hf-namespace",default="lerobot")
    p.add_argument("--repo-id"); p.add_argument("--budget-vision",type=int,default=40)
    p.add_argument("--budget-language",type=int,default=75); p.add_argument("--budget-action-head",type=int,default=1)
    p.add_argument("--suppression-beta", type=float, default=1.0)
    p.add_argument("--suppression-power", type=float, default=2.0)
    p.add_argument(
        "--update-threshold", type=float, default=0.0,
        help="CLASSP-style threshold on accumulated grad^2; 0 disables thresholding",
    )
    p.add_argument("--importance-decay",type=float,default=1.0)
    p.add_argument("--importance-max-batches",type=int,default=256)
    p.add_argument("--importance-batch-size",type=int,default=1,
                   help="Post-task importance batch size; 1 gives frame-level importance")
    p.add_argument("--epochs",type=int,default=3); p.add_argument("--batch-size",type=int,default=10)
    p.add_argument("--grad-accum",type=int,default=2); p.add_argument("--lr",type=float,default=5e-5)
    p.add_argument("--num-demos",type=int); return p.parse_args()

def repo_id(args):
    if args.repo_id: return args.repo_id
    suffix=REPOS[args.suite].split('/',1)[1]
    return f"{args.hf_namespace}/{suffix}"

def selection(model,args,out):
    previous=Path(args.init_checkpoint)/"selection.json"
    if previous.is_file():
        info=json.loads(previous.read_text()); selected=set(info["selected_tensors"]); found=set()
        for name,p in model.named_parameters():
            p.requires_grad_(name in selected)
            if name in selected: found.add(name)
        if selected-found: raise KeyError(f"Missing selected tensors: {sorted(selected-found)[:10]}")
        info["selection_source"]=str(previous)
    else:
        budget={"vision":args.budget_vision,"language":args.budget_language,"action_head":args.budget_action_head}
        picked=select_low_kappa_tensors(model,budget,TensorGroupPatterns()); count=apply_selection(model,picked)
        info={"method":"kappatune_risk_importance","budget_per_group":budget,
              "selected_tensors":sorted(picked.trainable),"selection_summary":summarize_selection(picked),
              "n_trainable":count,"selection_source":"computed_once_from_initial_checkpoint"}
    (out/"selection.json").write_text(json.dumps(info,indent=2)); return info

def wrap_collate(base):
    def collate(samples):
        risk=torch.stack([x["dynamics_risk"] for x in samples]); clean=[]
        for x in samples:
            y=dict(x); y.pop("dynamics_risk"); clean.append(y)
        batch=base(clean); batch["dynamics_risk"]=risk; return batch
    return collate

def main():
    a=parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    model,processor=load_model_and_processor(a.init_checkpoint); info=selection(model,a,out)
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    print(info.get("selection_summary","Loaded fixed KappaTune selection")); print(f"Trainable tensors: {len(named)}")
    old_imp,_old_ref=load_state(a.init_checkpoint)
    rid=repo_id(a); dataset=load_suite_dataset(SuiteSpec(name=a.suite,repo_id=rid,num_demos=a.num_demos))
    dataset=RiskAugmentedDataset(dataset,a.risk_scores); q01,q99=compute_action_bounds(dataset.dataset)
    collate=wrap_collate(make_collate_fn(processor,q01,q99))
    loader=DataLoader(dataset,batch_size=a.batch_size,shuffle=True,num_workers=4,pin_memory=True,
                      drop_last=True,collate_fn=collate)
    opt=DynamicsRiskSuppressedAdamW(
        named_parameters=named,
        importance=old_imp,
        lr=a.lr,
        beta=a.suppression_beta,
        power=a.suppression_power,
        threshold=a.update_threshold,
    )
    print(json.dumps({"adjustment_suppression": opt.get_suppression_report()}, indent=2), flush=True)
    model.train(); opt.zero_grad(set_to_none=True)
    micro=steps=0; started=time.time(); log=out/"loss_history.jsonl"
    if log.exists(): log.unlink()
    for epoch in range(a.epochs):
        for bi,batch in enumerate(loader):
            batch.pop("dynamics_risk"); batch={k:(v.to("cuda",non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}
            imitation=model(**batch).loss
            # No reference/displacement penalty. Historical importance acts only
            # through the per-tensor AdamW learning-rate denominator.
            total=imitation
            (total/a.grad_accum).backward(); micro+=1
            if micro%a.grad_accum==0 or bi+1==len(loader):
                threshold_report = opt.apply_threshold()
                torch.nn.utils.clip_grad_norm_((p for _,p in named),1.0)
                opt.step(); opt.zero_grad(set_to_none=True); steps+=1
            if micro==1 or micro%5==0:
                row={"epoch":epoch+1,"micro_step":micro,"optimizer_step":steps,
                     "imitation_loss":float(imitation.detach()),
                     "total_loss":float(total.detach()),
                     "threshold_kept_fraction":float(opt.get_suppression_report()["kept_fraction"]),
                     "elapsed_seconds":time.time()-started}
                with log.open("a") as f: f.write(json.dumps(row)+"\n")
                print(json.dumps(row),flush=True)
    importance_loader=DataLoader(dataset,batch_size=a.importance_batch_size,shuffle=False,num_workers=4,
                                 pin_memory=True,drop_last=False,collate_fn=collate)
    new_imp,new_ref,report=estimate(model,importance_loader,max_batches=a.importance_max_batches)
    merged=merge(old_imp,new_imp,decay=a.importance_decay)
    report.update({
        "suite": a.suite, "repo_id": rid,
        "protection_method": "dynamics-risk-weighted-adjustment-suppression",
        "suppression_beta": a.suppression_beta,
        "suppression_power": a.suppression_power,
        "update_threshold": a.update_threshold,
        "threshold_relaxed": a.update_threshold == 0.0,
        "suppression_report": opt.get_suppression_report(),
        "importance_decay": a.importance_decay,
        "importance_batch_size": a.importance_batch_size,
        "risk_scores": a.risk_scores,
    })
    save(out,merged,new_ref,report)
    stats=Path(a.init_checkpoint)/"action_stats.json"; registry=json.loads(stats.read_text()) if stats.is_file() else {}
    registry[a.suite]={"q01":np.asarray(q01).tolist(),"q99":np.asarray(q99).tolist()}
    (out/"action_stats.json").write_text(json.dumps(registry,indent=2)); model.save_pretrained(out); processor.save_pretrained(out)
if __name__=="__main__": main()
