#!/usr/bin/env python

""" Run celery workers on each GPU """

import torch
import subprocess
import os

n_gpus = torch.cuda.device_count()
print(f"Running {n_gpus} workers")

output_parent_dir = os.environ.get("OUTPUT_DIR", "output")
print("Output parent dir:", output_parent_dir)

procs = []
for i in range(n_gpus):
    out_folder = f"{output_parent_dir}/{i}"
    os.makedirs(out_folder, exist_ok=True)
    cmd = f"CUDA_VISIBLE_DEVICES={i} OUTPUT_DIR={out_folder} python -m common.diffdock"
    print(f"Running {cmd}")
    p = subprocess.Popen(cmd, shell=True)
    procs.append(p)

for p in procs:
    p.wait()