#!/usr/bin/env python3
"""One-stage continual OpenVLA trainer with memory-conscious online diagonal EWC."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data_utils import SuiteSpec, load_suite_dataset
from ewc_state import estimate_empirical_fisher, load_cpu, merge_online_in_place, move_state_to_parameters, penalty as ewc_penalty, save as save_ewc
from kappatune_vla import TensorGroupPatterns, apply_selection, select_low_kappa_tensors, summarize_selection
from train_continual import compute_action_bounds, load_model_and_processor, make_collate_fn

REPOS={"libero_spatial":"lerobot/libero_spatial_image","libero_object":"lerobot/libero_object_image","libero_goal":"lerobot/libero_goal_image","libero_long":"lerobot/libero_10_image","libero_90":"lerobot/libero_90_image"}

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--suite",required=True,choices=list(REPOS)); p.add_argument("--init-checkpoint",required=True); p.add_argument("--out-dir",required=True)
    p.add_argument("--hf-namespace",default="lerobot"); p.add_argument("--repo-id"); p.add_argument("--selection-mode",choices=["kappatune","full"],default="kappatune")
    p.add_argument("--budget-vision",type=int,default=40); p.add_argument("--budget-language",type=int,default=75); p.add_argument("--budget-action-head",type=int,default=1)
    p.add_argument("--ewc-lambda",type=float,default=1.0); p.add_argument("--ewc-decay",type=float,default=1.0)
    p.add_argument("--fisher-max-batches",type=int,default=256); p.add_argument("--fisher-batch-size",type=int,default=1)
    p.add_argument("--epochs",type=int,default=4); p.add_argument("--batch-size",type=int,default=10); p.add_argument("--grad-accum",type=int,default=2)
    p.add_argument("--lr",type=float,default=5e-5); p.add_argument("--weight-decay",type=float,default=0.01); p.add_argument("--num-demos",type=int); p.add_argument("--num-workers",type=int,default=4)
    p.add_argument("--fisher-only",action="store_true",help="Load trained model from --out-dir and compute/save only EWC state")
    return p.parse_args()

def repo_id(a):
    if a.repo_id: return a.repo_id
    return f"{a.hf_namespace}/{REPOS[a.suite].split('/',1)[1]}"

def configure_selection(model,a,out):
    previous=Path(a.init_checkpoint)/"selection.json"
    if previous.is_file():
        info=json.loads(previous.read_text()); mode=info.get("selection_mode","kappatune")
        if mode!=a.selection_mode: raise ValueError(f"Selection mode changed: {mode} -> {a.selection_mode}")
        selected=set(info["selected_tensors"]); found=set()
        for name,param in model.named_parameters():
            param.requires_grad_(name in selected)
            if name in selected: found.add(name)
        missing=selected-found
        if missing: raise KeyError(f"Missing selected tensors: {sorted(missing)[:10]}")
    elif a.selection_mode=="full":
        selected=set(); count=0
        for name,param in model.named_parameters(): param.requires_grad_(True); selected.add(name); count+=param.numel()
        info={"method":"ewc","selection_mode":"full","selected_tensors":sorted(selected),"n_trainable":count,"selection_source":"all_model_parameters"}
    else:
        budget={"vision":a.budget_vision,"language":a.budget_language,"action_head":a.budget_action_head}
        picked=select_low_kappa_tensors(model,budget,TensorGroupPatterns()); count=apply_selection(model,picked)
        info={"method":"ewc","selection_mode":"kappatune","budget_per_group":budget,"selected_tensors":sorted(picked.trainable),"selection_summary":summarize_selection(picked),"n_trainable":count}
    if not info["selected_tensors"]: raise RuntimeError("No trainable tensors selected")
    (out/"selection.json").write_text(json.dumps(info,indent=2)); return info

def main():
    a=parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    if a.fisher_only:
        marker=out/"TRAINING_WEIGHTS_SAVED"
        if not marker.is_file(): raise FileNotFoundError(f"Cannot use --fisher-only: {marker} is missing")
        model_source=str(out)
    else: model_source=a.init_checkpoint
    model,processor=load_model_and_processor(model_source)
    if a.fisher_only:
        sp=out/"selection.json"
        if not sp.is_file(): raise FileNotFoundError(f"Missing {sp}")
        info=json.loads(sp.read_text()); selected=set(info["selected_tensors"])
        for name,param in model.named_parameters(): param.requires_grad_(name in selected)
    else: info=configure_selection(model,a,out)
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]; n_params=sum(p.numel() for _,p in named)
    print(json.dumps({"fisher_only":a.fisher_only,"selection_mode":a.selection_mode,"trainable_tensors":len(named),"trainable_parameters":n_params},indent=2),flush=True)

    old_fisher_cpu,old_anchor_cpu,old_meta=load_cpu(a.init_checkpoint)
    old_fisher,old_anchor=move_state_to_parameters(named,old_fisher_cpu,old_anchor_cpu)
    rid=repo_id(a); dataset=load_suite_dataset(SuiteSpec(name=a.suite,repo_id=rid,num_demos=a.num_demos))
    q01,q99=compute_action_bounds(dataset); collate=make_collate_fn(processor,q01,q99)

    if not a.fisher_only:
        loader=DataLoader(dataset,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,pin_memory=True,drop_last=True,collate_fn=collate)
        opt=torch.optim.AdamW([p for _,p in named],lr=a.lr,weight_decay=a.weight_decay)
        model.train(); opt.zero_grad(set_to_none=True); micro=steps=0; started=time.time(); log=out/"loss_history.jsonl"
        if log.exists(): log.unlink()
        for epoch in range(a.epochs):
            for bi,batch in enumerate(loader):
                batch={k:(v.to("cuda",non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}
                imitation=model(**batch).loss; regularizer=ewc_penalty(named,old_fisher,old_anchor,a.ewc_lambda); total=imitation+regularizer
                (total/a.grad_accum).backward(); micro+=1
                if micro%a.grad_accum==0 or bi+1==len(loader):
                    torch.nn.utils.clip_grad_norm_((p for _,p in named),1.0); opt.step(); opt.zero_grad(set_to_none=True); steps+=1
                if micro==1 or micro%5==0:
                    row={"epoch":epoch+1,"micro_step":micro,"optimizer_step":steps,"imitation_loss":float(imitation.detach()),"ewc_penalty":float(regularizer.detach()),"total_loss":float(total.detach()),"elapsed_seconds":time.time()-started}
                    with log.open("a") as h: h.write(json.dumps(row)+"\n")
                    print(json.dumps(row),flush=True)
        print("Training complete; saving model before Fisher estimation",flush=True)
        model.save_pretrained(out); processor.save_pretrained(out); (out/"TRAINING_WEIGHTS_SAVED").touch()
    else:
        print("Fisher-only resume from saved stage weights",flush=True)

    fisher_loader=DataLoader(dataset,batch_size=a.fisher_batch_size,shuffle=False,num_workers=a.num_workers,pin_memory=True,drop_last=False,collate_fn=collate)
    new_fisher,new_anchor,report=estimate_empirical_fisher(model,fisher_loader,named,a.fisher_max_batches)
    merged=merge_online_in_place(old_fisher_cpu,new_fisher,a.ewc_decay)
    report.update({"suite":a.suite,"repo_id":rid,"method":"online_diagonal_ewc","selection_mode":a.selection_mode,"ewc_lambda":a.ewc_lambda,"ewc_decay":a.ewc_decay,"fisher_batch_size":a.fisher_batch_size,"fisher_max_batches":a.fisher_max_batches,"trainable_tensors":len(named),"trainable_parameters":n_params,"epochs":a.epochs,"lr":a.lr,"previous_state_loaded":bool(old_fisher_cpu)})
    save_ewc(out,merged,new_anchor,report)
    prev=Path(a.init_checkpoint)/"action_stats.json"; registry=json.loads(prev.read_text()) if prev.is_file() else {}
    registry[a.suite]={"q01":np.asarray(q01).tolist(),"q99":np.asarray(q99).tolist()}; (out/"action_stats.json").write_text(json.dumps(registry,indent=2))
    (out/"FISHER_COMPLETE").touch(); print(f"Saved EWC state at {out}",flush=True)
if __name__=="__main__": main()
