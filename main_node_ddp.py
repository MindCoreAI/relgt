import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Dict
import wandb

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import pynvml  # GPU utilization stats (NVIDIA only)
except Exception:  # pragma: no cover
    pynvml = None

from torch.nn import BCEWithLogitsLoss, L1Loss
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

from torch_frame._stype import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

# within this project
from model import RelGT
from utils import GloveTextEmbedding, RelGTTokens

torch.autograd.set_detect_anomaly(True)

############################
# 1. Parse arguments
############################
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-f1")
parser.add_argument("--task", type=str, default="driver-top3")
parser.add_argument("--precompute", action="store_true", default=False)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--warmup_steps", type=int, default=1000)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=512)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--gt_conv_type", type=str, default="full")
parser.add_argument("--ablate", type=str, default="none")
parser.add_argument("--gnn_pe_dim", type=int, default=0)
parser.add_argument("--num_neighbors", type=int, default=300)
parser.add_argument("--num_centroids", type=int, default=4096)
parser.add_argument("--ff_dropout", type=float, default=0.1)
parser.add_argument("--attn_dropout", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=0.00001)
parser.add_argument("--temporal_strategy", type=str, default="uniform")
parser.add_argument("--pos_enc", type=str, default="none")
parser.add_argument("--max_degree", type=int, default=10000)
parser.add_argument("--pos_enc_dim", type=int, default=128)
parser.add_argument("--max_steps_per_epoch", type=int, default=3000)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out_dir", type=str, default="results/debug")
parser.add_argument("--run_name", type=str, default="debug")
parser.add_argument('--model_parameters', type=int, default=0, help='Number of model parameters')

# --- Training performance knobs (CUDA) ---
parser.add_argument(
    "--amp",
    action="store_true",
    default=False,
    help="Enable CUDA mixed precision (torch.cuda.amp).",
)
parser.add_argument(
    "--grad_accum_steps",
    type=int,
    default=1,
    help="Accumulate gradients over N micro-batches before optimizer step.",
)

parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("~/.cache/relbench_examples"),
)
parser.add_argument("--train_stage", type=str, default="finetune", choices=["finetune"])

args = parser.parse_args()

############################
# 2. Initialize DDP and set device
############################
# NOTE: Original code assumes multi-GPU NVIDIA (NCCL + CUDA).
# For CPU-only or Apple Silicon, fall back to GLOO and CPU/MPS.
backend = "nccl" if torch.cuda.is_available() else "gloo"

# If launched with torchrun, the env vars (RANK/WORLD_SIZE/LOCAL_RANK/MASTER_*) are set.
# Otherwise, fall back to a single-process process-group so the rest of the code (e.g.,
# DistributedSampler) can still run.
if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
    dist.init_process_group(backend=backend)
else:
    dist.init_process_group(
        backend=backend,
        init_method="file:///tmp/relgt_dist_init",
        rank=0,
        world_size=1,
    )

local_rank = int(os.environ.get("LOCAL_RANK", 0))

if torch.cuda.is_available():
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
elif (
    os.environ.get("RELGT_USE_MPS", "0") == "1"
    and getattr(torch.backends, "mps", None) is not None
    and torch.backends.mps.is_available()
):
    # MPS can be flaky for some ops; enable only when explicitly requested.
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Only the main process (rank 0) initializes wandb and prints logs.
if local_rank == 0:
    args.run_name = f"{args.dataset}-{args.task}-{args.run_name}"

def init_gpu_utilization(device_index):
    if (pynvml is None) or (not torch.cuda.is_available()):
        return None
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    return handle


def get_gpu_stats(handle, device):
    if handle is None:
        return None, None, None
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    gpu_util = util.gpu
    mem_allocated = torch.cuda.memory_allocated(device) / 1024**2
    mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
    return gpu_util, mem_allocated, mem_reserved

print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

gpu_handle = init_gpu_utilization(local_rank)

############################
# 3. Load dataset, task, and prepare data
############################
dataset: Dataset = get_dataset(args.dataset, download=True)
task: EntityTask = get_task(args.dataset, args.task, download=True)

stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
try:
    with open(stypes_cache_path, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)
except FileNotFoundError:
    col_to_stype_dict = get_stype_proposal(dataset.get_db())
    Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stypes_cache_path, "w") as f:
        json.dump(col_to_stype_dict, f, indent=2, default=str)

data, col_stats_dict = make_pkey_fkey_graph(
    dataset.get_db(),
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device), batch_size=256
    ),
    cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)

data = {
    split: RelGTTokens(
        data=data, 
        task=task,
        K=args.num_neighbors, 
        split=split, 
        undirected=True, 
        precompute=args.precompute,
        precomputed_dir=f"{args.cache_dir}/precomputed/{args.dataset}/{args.task}",
        num_workers=args.num_workers,
        train_stage=args.train_stage)
        for split in ["train", "val", "test"]
    }

############################
# 4. Create DataLoaders (with a DistributedSampler for training)
############################
train_sampler = DistributedSampler(data["train"], shuffle=True, seed=args.seed)
loader_train = DataLoader(
    data["train"], 
    batch_size=args.batch_size, 
    sampler=train_sampler,
    collate_fn=data["train"].collate,
    num_workers=args.num_workers,
    persistent_workers=args.num_workers > 0,
    pin_memory=True)

val_sampler = DistributedSampler(data["val"], shuffle=False, seed=args.seed, drop_last=False)
loader_val = DataLoader(
    data["val"],
    batch_size=args.batch_size,
    sampler=val_sampler,
    # shuffle=False,
    collate_fn=data["val"].collate,
    num_workers=args.num_workers,
    persistent_workers=(args.num_workers > 0),
    pin_memory=True
)

test_sampler = DistributedSampler(data["test"], shuffle=False, seed=args.seed, drop_last=False)
loader_test = DataLoader(
    data["test"],
    batch_size=args.batch_size,
    sampler=test_sampler,
    # shuffle=False,
    collate_fn=data["test"].collate,
    num_workers=args.num_workers,
    persistent_workers=(args.num_workers > 0),
    pin_memory=True
)


loader_dict: Dict[str, DataLoader] = {"train": loader_train, "val": loader_val, "test": loader_test}

############################
# 5. Set up the task-specific settings
############################
clamp_min, clamp_max = None, None
if task.task_type == TaskType.BINARY_CLASSIFICATION:
    out_channels = 1
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "roc_auc"
    higher_is_better = True
elif task.task_type == TaskType.REGRESSION:
    out_channels = 1
    loss_fn = L1Loss()
    tune_metric = "mae"
    higher_is_better = False
    train_table = task.get_table("train")
    clamp_min, clamp_max = np.percentile(
        train_table.df[task.target_col].to_numpy(), [2, 98]
    )
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    out_channels = task.num_labels
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "multilabel_auprc_macro"
    higher_is_better = True
else:
    raise ValueError(f"Task type {task.task_type} is unsupported")

############################
# 6. Build and wrap the model in DDP
############################
model = RelGT(
    num_nodes=data["train"].data.num_nodes,
    max_neighbor_hop=data["train"].max_neighbor_hop,
    node_type_map=data["train"].node_type_to_index,
    col_names_dict={node_type: data["train"].data[node_type].tf.col_names_dict 
                    for node_type in data["train"].data.node_types},
    col_stats_dict=col_stats_dict,
    local_num_layers=args.num_layers,
    channels=args.channels,
    out_channels=out_channels,
    global_dim=args.channels//2,
    heads=args.num_heads,
    ff_dropout=args.ff_dropout,
    attn_dropout=args.attn_dropout,
    conv_type=args.gt_conv_type,
    ablate=args.ablate,
    gnn_pe_dim=args.gnn_pe_dim,
    num_centroids=args.num_centroids,
    sample_node_len=args.num_neighbors,
    args=args,
).to(device)

# Before DDP initialization, cast problematic tensors
for name, param in model.named_parameters():
    if param.dtype == torch.int16:
        param.data = param.data.to(torch.int64)

for name, buf in model.named_buffers():
    if buf.dtype == torch.int16:
        buf.data = buf.data.to(torch.int64)

world_size = dist.get_world_size()

# SyncBatchNorm/DDP are only needed for multi-process training.
if world_size > 1:
    if device.type == "cuda":
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        model = DDP(model, find_unused_parameters=True)

if local_rank == 0:
    print(model)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
if local_rank == 0:
    print(f"Total model parameters: {total_params}")
args.model_parameters = total_params

if local_rank == 0:
    if os.environ.get("WANDB_MODE") != "disabled" and os.environ.get("WANDB_API_KEY"):
        wandb.init(project="rel-gt-expts", name=args.run_name, config=vars(args))

output_path = os.path.join(args.out_dir, args.dataset, args.task)
os.makedirs(output_path, exist_ok=True)

world_size = dist.get_world_size()
base_lr = args.lr * world_size
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)

use_amp = bool(args.amp and device.type == "cuda")
if args.grad_accum_steps < 1:
    raise ValueError("--grad_accum_steps must be >= 1")

scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

global_step = 0

############################
# 7. Training and Evaluation Loops
############################
def train_supervised(epoch) -> float:
    global global_step
    model.train()
    loss_accum = count_accum = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)

    train_sampler.set_epoch(epoch)

    # Gradient accumulation: delay optimizer step until every N micro-batches.
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(
        tqdm(loader_dict["train"], total=total_steps, desc="Train"), start=1
    ):
        # Move tensors to the proper device.
        neighbor_types = batch["neighbor_types"].to(device)
        node_indices = batch["node_indices"].to(device)
        neighbor_hops = batch["neighbor_hops"].to(device)
        neighbor_times = batch["neighbor_times"].to(device)
        edge_index = batch["edge_index"].to(device)
        batch_vec = batch["batch"].to(device)

        grouped_tf_dict = {
            "grouped_tfs": batch["grouped_tfs"],
            "grouped_indices": batch["grouped_indices"],
            "flat_batch_idx": batch["flat_batch_idx"],
            "flat_nbr_idx": batch["flat_nbr_idx"],
        }
        # MPS backend does not support float64; force labels to float32.
        labels = batch["labels"].to(device=device, dtype=torch.float32)

        # Forward (optionally in AMP autocast). Compute the loss in fp32 for stability.
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(
                neighbor_types,
                node_indices,
                neighbor_hops,
                neighbor_times,
                grouped_tf_dict,
                edge_index=edge_index,
                batch=batch_vec,
            )
            pred = pred.view(-1) if pred.size(1) == 1 else pred

        # Keep loss in fp32 even under AMP (BCEWithLogits can overflow / produce NaNs).
        pred_fp32 = pred.float()
        # Sanitize + clamp logits to avoid inf/nan in BCE.
        pred_fp32 = torch.nan_to_num(pred_fp32, nan=0.0, posinf=30.0, neginf=-30.0)
        if task.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION]:
            pred_fp32 = pred_fp32.clamp(min=-30.0, max=30.0)
        loss = loss_fn(pred_fp32, labels)
        loss = loss / args.grad_accum_steps

        # Skip non-finite loss (can happen under AMP). Clear grads and continue.
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            if local_rank == 0:
                print("[WARN] Non-finite loss encountered; skipping step")
            continue

        # Backward
        scaler.scale(loss).backward()

        # Logging values (de-scale for readability)
        loss_value = (loss.detach().item()) * args.grad_accum_steps

        loss_accum += loss_value * pred.size(0)
        count_accum += pred.size(0)

        should_step = (step % args.grad_accum_steps == 0) or (step >= args.max_steps_per_epoch)
        if should_step:
            # Unscale before grad clipping.
            if use_amp:
                scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            gpu_util, mem_allocated, mem_reserved = get_gpu_stats(gpu_handle, device)
            # Only rank 0 logs training metrics.
            if local_rank == 0 and wandb.run is not None:
                wandb.log(
                    {
                        "train_loss": loss_value,
                        "global_step": global_step,
                        "lr": optimizer.param_groups[0]["lr"],
                        "gpu_util_percent": gpu_util,
                        "gpu_mem_allocated_MB": mem_allocated,
                        "gpu_mem_reserved_MB": mem_reserved,
                        "amp": int(use_amp),
                        "grad_accum_steps": args.grad_accum_steps,
                    }
                )

        global_step += 1

        if step >= args.max_steps_per_epoch:
            break

    return loss_accum / count_accum if count_accum > 0 else float("inf")

@torch.no_grad()
def test(loader: DataLoader, eval_model, epoch, desc) -> np.ndarray:
    if loader.sampler is not None and hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)
        
    eval_model.eval()
    pred_list = []
    idx_list = []
    
    for batch in tqdm(loader, desc=desc, disable=(local_rank != 0)):
        neighbor_types = batch["neighbor_types"].to(device)
        node_indices = batch["node_indices"].to(device)
        neighbor_hops = batch["neighbor_hops"].to(device)
        neighbor_times = batch["neighbor_times"].to(device)
        edge_index = batch["edge_index"].to(device)
        batch_vec = batch["batch"].to(device)
        
        grouped_tf_dict = {
            'grouped_tfs': batch['grouped_tfs'],
            'grouped_indices': batch['grouped_indices'],
            'flat_batch_idx': batch['flat_batch_idx'],
            'flat_nbr_idx': batch['flat_nbr_idx']
        }
        pred = eval_model(
            neighbor_types,
            node_indices,
            neighbor_hops,
            neighbor_times,
            grouped_tf_dict,
            edge_index=edge_index,
            batch=batch_vec
        )
        if task.task_type == TaskType.REGRESSION:
            pred = torch.clamp(pred, clamp_min, clamp_max)
        if task.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION]:
            pred = torch.sigmoid(pred)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu().numpy())
        idx_list.append(batch["global_idx"].cpu().numpy())
    
    # Concatenate local predictions & indices
    local_preds = np.concatenate(pred_list, axis=0) if pred_list else np.array([])
    local_idxs  = np.concatenate(idx_list,  axis=0) if idx_list  else np.array([])

    # Gather on rank 0
    gathered = [None for _ in range(world_size)] if local_rank == 0 else None
    dist.gather_object((local_idxs, local_preds), object_gather_list=gathered, dst=0)

    if local_rank == 0:
        all_preds = np.full((len(loader.dataset),), -100.0)
        for i in range(world_size):
            g_idx, g_pred = gathered[i]
            for idx, pred in zip(g_idx, g_pred):
                all_preds[idx] = pred
        return all_preds
    else:
        return None

if args.train_stage == "finetune":
    # Supervised Finetuning Stage:
    best_val_metric = -math.inf if higher_is_better else math.inf
    state_dict = None

    for epoch in range(1, args.epochs + 1):
        # use supervised training loop.
        train_loss = train_supervised(epoch)
        # scheduler.step()
        
        dist.barrier()
        eval_model = model.module if hasattr(model, "module") else model
        
        # Run evaluation on the validation set.
        val_pred = test(loader_dict["val"], eval_model=eval_model, epoch=epoch, desc="Val")
        if local_rank == 0:
            val_metrics = task.evaluate(val_pred, task.get_table("val"))
            print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}")
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch,
                    "epoch_train_loss": train_loss,
                    **{f"val_{k}": v for k, v in val_metrics.items()}
                })
            
            if (higher_is_better and val_metrics[tune_metric] >= best_val_metric) or (
                not higher_is_better and val_metrics[tune_metric] <= best_val_metric
            ):
                best_val_metric = val_metrics[tune_metric]
                base_model = model.module if hasattr(model, "module") else model
                state_dict = copy.deepcopy(base_model.state_dict())
                torch.save(state_dict, os.path.join(output_path, "finetuned.pt"))
        dist.barrier()

    if local_rank == 0 and state_dict is not None:
        base_model = model.module if hasattr(model, "module") else model
        base_model.load_state_dict(state_dict)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    for buf in model.buffers():
        dist.broadcast(buf.data, src=0)
    dist.barrier()

    # Final evaluation after finetuning:
    base_model = model.module if hasattr(model, "module") else model
    final_val_preds = test(loader_dict["val"], eval_model=base_model, epoch=0, desc="Val")
    final_test_preds = test(loader_dict["test"], eval_model=base_model, epoch=0, desc="Test")

    if local_rank == 0:
        val_metrics = task.evaluate(final_val_preds, task.get_table("val"))
        print(f"Best Val metrics: {val_metrics}")

        test_metrics = task.evaluate(final_test_preds)
        print(f"Best Test metrics: {test_metrics}")

        best_metrics_dict = {
            "val_metrics": val_metrics,
            "test_metrics": test_metrics
        }
        file_path = os.path.join(output_path, str(args.seed) + ".json")
        with open(file_path, "w") as f:
            json.dump(best_metrics_dict, f, indent=4)
    
    if local_rank == 0:
        print(f"[{args.train_stage.capitalize()} Stage] Training complete. No supervised evaluation performed.")


############################
# 8. Cleanup
############################
dist.destroy_process_group()