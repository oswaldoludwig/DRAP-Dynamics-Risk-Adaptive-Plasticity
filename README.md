# DRAP: Dynamics-Risk Adaptive Plasticity

**Dynamics-Risk Adaptive Plasticity for Continual Learning in Physical AI**

This repository contains an OpenVLA/LIBERO continual-learning implementation that combines offline estimates of local task dynamics with risk-weighted parameter-importance tracking and per-tensor update suppression. The goal is to reduce interference with parameters associated with historically important, high-dynamics-risk data while adapting to a sequence of robot manipulation suites.

> **Paper:** [arXiv:2611.05029](https://arxiv.org/abs/2611.05029) · [PDF](https://arxiv.org/pdf/2611.05029)
>
> This README describes the implementation currently present in this repository. For the paper's formal definitions, experimental protocol, and reported results, consult the paper.

## Method at a glance

The DRAP training path is implemented by `train_continual_kappa_risk_suppression.py` and its supporting modules:

1. **Estimate dynamics risk offline.** `compute_libero_risk_scores.py` fits local ridge-regression models of the form `z[t+1] = A[t] z[t] + B[t] u[t] + c[t]` over demonstration windows. The largest singular value of the fitted state-transition matrix is used as a per-frame score. Scores are normalized by the suite median and clipped to a configurable range. This calculation uses demonstration state/action data, not a simulator or policy rollout.
2. **Track risk-weighted parameter importance.** After a task stage, `risk_weighted_tensor_importance.py` estimates frame-level squared-gradient importance using the precomputed risk scores and merges it with importance from previous stages.
3. **Suppress selected tensor updates.** `dynamics_risk_adjustment_suppression.py` uses cumulative importance to set a learning-rate scale for each trainable tensor:

   `lr_tensor = base_lr / (1 + beta * normalized_importance_tensor) ** (1 / power)`

   An optional gradient-squared threshold can additionally mask small updates. A threshold of `0` disables this masking, which is the default in the supplied training wrapper.
4. **Train sequentially and evaluate retention.** Each stage starts from the preceding checkpoint. `eval_forgetting.py` evaluates checkpoints across LIBERO suites and reports a success-rate matrix and forgetting relative to earlier continual-learning stages.

The training wrapper also uses KappaTune's low-kappa tensor selection to limit which model tensors are trainable; this is separate from the dynamics-risk-based suppression mechanism. The repository includes `train_continual.py` for full fine-tuning, LoRA, and KappaTune comparison paths.

## Repository contents

| File | Purpose |
| --- | --- |
| `compute_libero_risk_scores.py` | Offline per-frame dynamics-risk estimation for LIBERO demonstration data. |
| `risk_regularization.py` | Loads risk scores and attaches them to dataset samples by persistent frame index. |
| `risk_weighted_tensor_importance.py` | Estimates and merges risk-weighted tensor importance across stages. |
| `dynamics_risk_adjustment_suppression.py` | AdamW optimizer with per-tensor learning-rate scaling and optional thresholding. |
| `train_continual_kappa_risk_suppression.py` | One sequential-training stage using KappaTune selection and DRAP suppression. |
| `train_continual.py` | Baseline training paths (`full`, `lora`, and `kappatune`). |
| `eval_forgetting.py` | Closed-loop LIBERO evaluation and forgetting-matrix generation. |
| `train_vla_kappa_risk_suppression.sh` | Multi-stage risk-score and training orchestration wrapper. |
| `run_eval_forgetting.sh` | Cluster-oriented evaluation wrapper. |
| `kappatune_vla.def`, `kappatune_vla_eval.def` | Singularity/Apptainer container definitions for training and evaluation environments. |

## Requirements and setup

The code targets GPU training and uses PyTorch, Transformers/OpenVLA, LeRobot, and LIBERO/MuJoCo. The checked-in Singularity definition files pin versions for a CUDA 12.6 environment; consult those files when preparing a compatible environment.

The shell wrappers are designed for a cluster workflow using **Singularity** and, for evaluation, **Slurm**. They expect a project directory, container image, cached Hugging Face files, and output directories. Set the paths and environment variables for your system before running them. In particular, replace the example/placeholder project paths in the wrappers and ensure all Python modules required by the training scripts are available in the container or on its Python path.

The repository's training code imports `data_utils` and `kappatune_vla`, and the container definitions refer to additional build-context files. Ensure those project dependencies and files are available in your deployment; the scripts and definitions may need adapting to your local layout. A GPU with sufficient memory for the chosen OpenVLA checkpoint is required. Access to the Hugging Face model and LIBERO LeRobot datasets may also require network access and, depending on your setup, an `HF_TOKEN`.

## Training workflow

The supplied orchestration script defaults to four sequential stages: `libero_spatial`, `libero_object`, `libero_goal`, and `libero_long` (the LIBERO 10-task suite). It first computes or reuses one risk-score file per suite, then trains each stage from the previous checkpoint.

After configuring the wrapper's paths and making it executable if needed, launch it inside an allocation with a visible GPU and Singularity installed:

```bash
export PROJECT_DIR=/path/to/your/project
export CONTAINER="$PROJECT_DIR/kappatune_vla.sif"
export RUN_ROOT="$PROJECT_DIR/runs/drap"
export RISK_ROOT="$PROJECT_DIR/risk_scores"

bash train_vla_kappa_risk_suppression.sh
```

Useful wrapper overrides include:

```bash
# Choose suites and training hyperparameters
SUITES=libero_spatial,libero_object EPOCHS=5 BATCH_SIZE=10 LR=5e-5 \
  bash train_vla_kappa_risk_suppression.sh

# Recompute the offline risk scores
RECOMPUTE_RISK=1 bash train_vla_kappa_risk_suppression.sh

# Disable the importance-based learning-rate suppression for a control run
SUPPRESSION_BETA=0 bash train_vla_kappa_risk_suppression.sh
```

The wrapper's defaults and all supported settings are defined in `train_vla_kappa_risk_suppression.sh`. The stage trainer can also be invoked directly; see its CLI for required arguments:

```bash
python train_continual_kappa_risk_suppression.py --help
```

A stage saves a model/processor checkpoint and bookkeeping artifacts such as `selection.json`, `action_stats.json`, `loss_history.jsonl`, `risk_tensor_importance.pt`, and `risk_tensor_importance_report.json`. The orchestration script marks completed stages with `TRAINING_COMPLETE` and can skip those stages on a subsequent run.

## Evaluation

`eval_forgetting.py` runs closed-loop policy evaluation in the LIBERO simulator. The shell wrapper evaluates the checkpoints in sequence and writes a JSON results matrix (by default, `results/<method>/forgetting_matrix.json`). It can optionally evaluate the zero-shot `openvla/openvla-7b` checkpoint as a baseline and save rollout videos.

Configure the evaluation wrapper's project and container paths, then run it in a suitable Slurm allocation. For a short diagnostic, run the evaluator's import check or simulator smoke test in the prepared environment:

```bash
python eval_forgetting.py --check-imports
python eval_forgetting.py --smoke-test --suites libero_spatial
```

Example direct evaluation of a trained checkpoint:

```bash
python eval_forgetting.py \
  --checkpoint runs/drap/stage0_spatial \
  --stage-name stage0_spatial \
  --suites libero_spatial \
  --results-json results/drap/forgetting_matrix.json
```

Print an existing results matrix with:

```bash
python eval_forgetting.py --show-results results/drap/forgetting_matrix.json
```

The evaluator uses fixed LIBERO task initial states and records per-suite success rates. Forgetting is computed as the best success rate at an earlier continual stage minus the current stage's success rate; the zero-shot baseline is not included in that forgetting calculation.

## Important configuration notes

- Risk-score files are suite-specific `.npz` files and are matched to samples using their persistent `index`. The risk dataset wrapper intentionally errors if that index is absent rather than silently aligning scores by position.
- The default risk window, ridge coefficient, minimum transitions, and clipping bounds are configurable in the training wrapper. Risk-score generation should use the same dataset version and indexing as training.
- The defaults use `SUPPRESSION_BETA=1`, `SUPPRESSION_POWER=2`, and `UPDATE_THRESHOLD=0`. Set `SUPPRESSION_BETA=0` for a no-suppression control; a positive update threshold enables the additional elementwise gradient mask.
- Checkpoint stage order matters: each continual stage should initialize from the preceding stage's output to measure sequential adaptation and forgetting.
- The scripts expect CUDA and use `bfloat16` for model execution. Container, CUDA driver, simulator, dataset, and model access must be validated on the target machine before a full run.

## Citation

If you use this implementation, please cite the associated paper:

```bibtex
@article{drap2026,
  title   = {Dynamics-Risk Adaptive Plasticity for Continual Learning in Physical AI},
  journal = {arXiv preprint arXiv:2611.05029},
  year    = {2026},
  url     = {https://arxiv.org/abs/2611.05029}
}
```

Please verify the paper's author list and preferred citation format on the [arXiv abstract page](https://arxiv.org/abs/2611.05029) before using this BibTeX entry in a publication.
