import os
import time
import threading
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import (
    is_ddp, get_rank, get_world_size,
    ddp_setup, ddp_barrier, ddp_cleanup, wrap_ddp,
    power_monitor,
)
# TODO: PretrainWorkload, SFTWorkload, MLLMWorkload, EvalWorkload
#       are not yet defined in LLM_toymodel.py — add them there.
from LLM_toymodel import *
import numpy as np
import matplotlib.pyplot as plt

def _to_numpy_power(power_samples):
    p = np.array(power_samples, dtype=np.float32)
    if p.ndim == 1:
        p_mat = p.reshape(-1, 1)
        p_total = p
    elif p.ndim == 2:
        p_mat = p
        p_total = p.sum(axis=1)
    else:
        raise ValueError(f"Unexpected power_samples shape: {p.shape}")
    return p_mat, p_total


def plot_power_with_phases(times, total_power, phase_times, phase_labels, title, save_path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, total_power, linewidth=1.0)

    phase_colors = {
        "pretrain": (0.2, 0.8, 0.2, 0.10),
        "sft": (0.2, 0.2, 0.9, 0.10),
        "mllm": (0.9, 0.2, 0.2, 0.10),
        "eval": (0.9, 0.6, 0.2, 0.10),
    }

    def base(lbl):
        return lbl.rsplit("_", 1)[0]

    i = 0
    shown = set()
    while i + 1 < len(phase_labels):
        ls = phase_labels[i]
        le = phase_labels[i + 1]
        ts = phase_times[i]
        te = phase_times[i + 1]
        if ls.endswith("_s") and le.endswith("_e") and base(ls) == base(le) and te > ts:
            name = base(ls)
            color = phase_colors.get(name, (0.5, 0.5, 0.5, 0.08))
            label = name if name not in shown else None
            ax.axvspan(ts, te, color=color, label=label)
            shown.add(name)
            i += 2
        else:
            i += 1

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Total GPU Power (W)")
    ax.set_title(title)
    if shown:
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

def main():
    rank, world = ddp_setup()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    pretrain = PretrainWorkload(batch_size=64, seq_len=512, device=device)
    sft = SFTWorkload(batch_size=16, seq_len=512, device=device)
    mllm = MLLMWorkload(batch_size=2, image_size=224, device=device)
    eval_task = EvalWorkload(batch_size=4, gen_len=64, device=device)

    t0 = time.time()
    stop_flag = {"stop": False}
    timestamps, power_samples = [], []
    phase_times, phase_labels = [], []

    monitor = None
    if rank == 0:
        monitor = threading.Thread(
            target=power_monitor,
            args=(0.05, stop_flag, timestamps, power_samples, t0),
            daemon=True
        )
        monitor.start()

    ddp_barrier()

    time.sleep(10)
    phase_times.append(time.time() - t0); phase_labels.append("pretrain_s")
    pretrain.run(n_steps=100)
    phase_times.append(time.time() - t0); phase_labels.append("pretrain_e")
    time.sleep(10)
    phase_times.append(time.time() - t0); phase_labels.append("sft_s")
    sft.run(n_steps=200)
    phase_times.append(time.time() - t0); phase_labels.append("sft_e")
    time.sleep(10)
    phase_times.append(time.time() - t0); phase_labels.append("mllm_s")
    mllm.run(n_steps=300)
    phase_times.append(time.time() - t0); phase_labels.append("mllm_e")
    time.sleep(10)
    phase_times.append(time.time() - t0); phase_labels.append("eval_s")
    eval_task.run(n_steps=80)
    phase_times.append(time.time() - t0); phase_labels.append("eval_e")

    ddp_barrier()

    if rank == 0:
        stop_flag["stop"] = True
        monitor.join()

        print(f"Collected {len(timestamps)} samples.")

    if rank == 0:
        times = np.array(timestamps, dtype=np.float32)
        p_mat, p_total = _to_numpy_power(power_samples)
        phase_t = np.array(phase_times, dtype=np.float32)
        phase_l = np.array(phase_labels, dtype=object)

        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "timestamps.npy"), times)
        np.save(os.path.join(out_dir, "power_per_gpu.npy"), p_mat)   # (N, ngpu) or (N,1)
        np.save(os.path.join(out_dir, "power_total.npy"), p_total)   # (N,)
        np.save(os.path.join(out_dir, "phase_times.npy"), phase_t)
        np.save(os.path.join(out_dir, "phase_labels.npy"), phase_l)

        fig_path = os.path.join(out_dir, "power_draw.png")
        plot_power_with_phases(
            times=times,
            total_power=p_total,
            phase_times=phase_t,
            phase_labels=phase_l,
            title="Power Draw with Workload Phases",
            save_path=fig_path
        )
    ddp_cleanup()

if __name__ == "__main__":
    main()
# torchrun --standalone --nproc_per_node=2 code/run.py
