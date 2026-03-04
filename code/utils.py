import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pynvml
import time
import threading
def get_rank():
    return int(os.environ["RANK"]) if is_ddp() else 0

def get_world_size():
    return int(os.environ["WORLD_SIZE"]) if is_ddp() else 1
def is_ddp():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ

def setup_ddp():
    if not is_ddp():
        return 0, 1
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    return rank, world

def wrap_ddp(model, device, enable_ddp=True):
    model = model.to(device)
    if enable_ddp and is_ddp():
        model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False, bucket_cap_mb=25, find_unused_parameters=True)
    return model

def init_nvml():
    pynvml.nvmlInit()
    ngpu = pynvml.nvmlDeviceGetCount()
    print(f"NVML init, {ngpu} GPUs detected")
    return ngpu


def read_power_all(ngpu):
    powers = []
    for i in range(ngpu):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetPowerUsage(h) # mW
        powers.append(info / 1000.0)
    return powers

def power_monitor(interval, stop_flag, out_times, out_powers, t0):
    ngpu = init_nvml()
    while not stop_flag["stop"]:
        now = time.time() - t0
        ps = read_power_all(ngpu)
        out_times.append(now)
        out_powers.append(ps)
        time.sleep(interval)
        if len(out_times) != len(out_powers):
            print('error in ', len(out_times))
    pynvml.nvmlShutdown()

