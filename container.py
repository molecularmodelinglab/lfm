#!/usr/bin/env python

from collections import defaultdict
from glob import glob
import os
import subprocess
import sys
from common.utils import CONFIG

VERSION = "0.0.1"

all_images = [item.split(".")[-1] for item in glob("docker/Dockerfile.*")]
aliases = {
    "all": all_images,
}


def compile_requirements():
    """Split the requirements.txt file into multiple files based on the Dockerfile used"""

    # first remove all the old files
    for item in glob("docker/requirements/*"):
        os.remove(item)

    req_dict = {}

    with open("docker/requirements.txt") as f:

        manager = None
        images = []

        for line in f:
            line = line.strip()
            if line == "":
                continue

            if line.startswith("##"):
                manager = line.split()[1]
                req_dict[manager] = defaultdict(set)

            elif line.startswith("#"):
                images = [image.strip(",") for image in line.split()[1:]]
                # make sure dev is included
                if "dev" not in images:
                    images.append("dev")
                # general is everything but those that are only for dev
                if "general" not in images and not (
                    "dev" in images and len(images) == 1
                ):
                    images.append("general")

            else:
                for image in images:
                    if image in aliases:
                        for alias in aliases[image]:
                            req_dict[manager][alias].add(line)
                    else:
                        req_dict[manager][image].add(line)

    for manager, cur_dict in req_dict.items():
        for image, reqs in cur_dict.items():
            out_file = f"docker/requirements/{manager}_{image}.txt"
            with open(out_file, "w") as f:
                for req in reqs:
                    f.write(req + "\n")


def build_image(image_name, tag):
    dockerfile = os.path.join("docker", f"Dockerfile.{image_name}")
    cmd = f"""
        docker build \
        -f {dockerfile} \
        --network=host -t {CONFIG.docker_prefix}/{image_name}:{tag} .
    """
    print(f"Running command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    cmd = f"docker tag {CONFIG.docker_prefix}/{image_name}:{tag} {CONFIG.docker_prefix}/{image_name}:latest"
    print(f"Running command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def upload_image(image_name):
    for tag in [VERSION, "latest"]:
        cmd = f"docker push {CONFIG.docker_prefix}/{image_name}:{tag}"
        print(f"Running command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

def upload_image_gcloud(image_name):
    for tag in [VERSION, "latest"]:
        new_image = f"us-east1-docker.pkg.dev/molecular-modelling-lab/rec-pmf/{image_name}:{tag}"
        cmd = f"docker tag {CONFIG.docker_prefix}/{image_name}:{tag} {new_image}"
        print(f"Running command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        cmd = f"docker push {new_image}"
        print(f"Running command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)


def run_image(image_name):
    """Run with lots of RAM and GPU access"""
    cmd = f"docker run --net=host --rm -it --memory='12g' --gpus all {CONFIG.docker_prefix}/{image_name}:latest"
    print(f"Running command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def run_interactive(image_name):
    """Run interactively"""
    username = "appuser" if image_name == "diffdock" else "mambauser"
    cmd = f"docker run --net=host -it --rm --memory='12g' --gpus all --volume .:/home/{username}/rec_pmf_net --entrypoint /bin/bash {CONFIG.docker_prefix}/{image_name}"
    print(f"Running command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    # always compile the requirements
    compile_requirements()
    
    actions = sys.argv[1].split(",")
    image_name = sys.argv[2]
    images = [image_name] if image_name not in aliases else aliases[image_name]


    for image_name in images:
        for action in actions:
            match action:
                case "build":
                    build_image(image_name, VERSION)
                case "upload":
                    upload_image(image_name)
                case "upload_gcloud":
                    upload_image_gcloud(image_name)
                case "run":
                    run_image(image_name)
                case "run_it":
                    run_interactive(image_name)
                case _:
                    raise ValueError(f"Unknown action: {action}")
