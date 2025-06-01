#!/usr/bin/env python

""" Run celery workers on each GPU """

import sys
import GPUtil
import os
import subprocess

if "--single" in sys.argv or os.environ.get("USE_SINGLE", "0") == "1":
    print("Running single worker")
    n_gpus = 1
elif "--cpus" in sys.argv or os.environ.get("USE_CPUS", "0") == "1":
    print("Parallelizing based on number of CPUs")
    n_gpus = os.cpu_count()
else:
    n_gpus = len(GPUtil.getGPUs())

print(f"Running {n_gpus} workers")

output_parent_dir = os.environ.get("OUTPUT_DIR", "output")
# the task to run. If celery, any celery task
task_name = os.environ.get("TASK_NAME", "celery")

procs = []
for i in range(n_gpus):
    gpu_id = "''" if "--cpus" in sys.argv else f"{i}"
    out_folder = f"{output_parent_dir}/{i}"

    if task_name == "celery":
        run_cmd = "celery -A common.all_tasks worker --loglevel=info --pool=solo"
    else:
        run_cmd = f"python -m common.all_tasks {task_name}"

    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} OUTPUT_DIR={out_folder} {run_cmd}"
    print(f"Running {cmd}")
    p = subprocess.Popen(cmd, shell=True)
    procs.append(p)

for p in procs:
    p.wait()