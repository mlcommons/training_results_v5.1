import torch
import os
import uuid
import pathlib
from dataclasses import dataclass


TORCHPROF_OUTPUT_DIR = os.getenv("TORCHPROF_OUTPUT_DIR", "/workspace/code/runs/torchprof")
PROF_OUTPUT_PATH = os.getenv("PROF_OUTPUT_PATH", "/workspace/code/profiler")

TORCHPROF_OUTPUT = os.getenv("TORCHPROF_OUTPUT", "csv_handler")
TORCHPROF_VERBOSE = os.getenv("TORCHPROF_VERBOSE", 0)
TORCHPROF_DEVICES = os.getenv("TORCHPROF_DEVICES", "GPU")
TORCHPROF_MAXROWS = os.getenv("TORCHPROF_MAXROWS", 100)
PROF_WARMUP_STEPS = int(os.getenv("PROF_WARMUP_STEPS", 2))
PROF_ACTIVE_STEPS = int(os.getenv("PROF_ACTIVE_STEPS", 2))
PROF_REPITIONS = int(os.getenv("PROF_REPITIONS", 0))


@dataclass
class TorchProfConfig:
    skip_first = 1
    wait = 0
    warmup = PROF_WARMUP_STEPS
    active = PROF_ACTIVE_STEPS
    repeat = PROF_REPITIONS


TOTAL_WARMUP_STEPS = TorchProfConfig.skip_first + \
    TorchProfConfig.wait + \
    TorchProfConfig.warmup

TOTAL_ACTIVE_STEPS = TorchProfConfig.active


def get_devices() -> list:
    devices = TORCHPROF_DEVICES.split(",")
    devices = [x.lower() for x in devices]
    devices_set = set()
    for device in devices:
        if device.lower() not in ['cpu', 'gpu']:
            raise ValueError(f"Invalid Device :{device}")
        if device.lower() == "gpu":
            devices_set.add(torch.profiler.ProfilerActivity.CUDA)
        elif device.lower() == "cpu":
            devices_set.add(torch.profiler.ProfilerActivity.CPU)
        
    devices_list = list(devices_set)

    return devices_list


def trace_handler(prof):
    pathlib.Path(TORCHPROF_OUTPUT_DIR).mkdir(exist_ok=True)
    save_path = f"{TORCHPROF_OUTPUT_DIR}/key_avg_{prof.step_num}_{uuid.uuid4()}.txt"
    print(f"Saving torchprof results at: {save_path}")
    with open(save_path, 'w') as f:
        output = prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=TORCHPROF_MAXROWS) 
        f.write(output)
        if TORCHPROF_VERBOSE:
            print(output)

    prof.export_chrome_trace(f"/{TORCHPROF_OUTPUT_DIR}/trace_{prof.step_num}_{uuid.uuid4()}.json")

def _get_rpd():
    pathlib.Path(PROF_OUTPUT_PATH).mkdir(exist_ok=True)
    from rpdTracerControl import rpdTracerControl
    rpdTracerControl.setFilename(name=f"{PROF_OUTPUT_PATH}/trace.rpd", append=True)
    prof = rpdTracerControl()
    return prof


def _get_torchprof():
    if TORCHPROF_OUTPUT == "csv_handler":
        output_handler = trace_handler
    elif TORCHPROF_OUTPUT == "tensorboard":
        output_handler = torch.profiler.tensorboard_trace_handler(TORCHPROF_OUTPUT_DIR)
    else:
        raise ValueError("Invalid Output Handler for TorchProf.")

    prof = torch.profiler.profile(
        activities=get_devices(), 
        schedule=torch.profiler.schedule(
            skip_first=TorchProfConfig.skip_first,
            wait=TorchProfConfig.wait,
            warmup=TorchProfConfig.warmup,
            active=TorchProfConfig.active, 
            repeat=TorchProfConfig.repeat
        ),
        # called each time the trace is ready at the end of each cycle.  
        on_trace_ready=output_handler,
        profile_memory=False, # adds extra overhead if True !!
        with_flops=False,
        with_stack=True, # adds extra overhead if True !! 
        record_shapes=False
    )

    return prof

def get_profiler(prof_type):
    if prof_type == 'rpd':
        return _get_rpd()
    elif prof_type == 'torchprof':
        return _get_torchprof()
    else:
        raise ValueError(f"Invalid profiler_type: {prof_type}")
    