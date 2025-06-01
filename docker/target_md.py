#!/usr/bin/env python

""" Run celery workers on each GPU """

import GPUtil
import subprocess

n_gpus = len(GPUtil.getGPUs())
print(f"Running {n_gpus} workers")

procs = []
for i in range(n_gpus):
    out_folder = f"output/{i}"
    cmd = f"CUDA_VISIBLE_DEVICES={i} OUTPUT_DIR={out_folder} celery -A common.md_sim worker --loglevel=info --pool=solo"
    print(f"Running {cmd}")
    p = subprocess.Popen(cmd, shell=True)
    procs.append(p)

for p in procs:
    p.wait()