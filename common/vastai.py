import json
import subprocess
import requests
import os
import pandas as pd

docker_creds = json.load(open("secrets/docker_creds.json"))
docker_username = docker_creds["username"]
docker_password = docker_creds["password"]

vast_fname = os.path.expanduser("~/.vast_api_key")
with open(vast_fname, 'r') as f:
    vast_api_key = f.read().strip()

def get_user_info():
    headers = {
        "Authorization": f"Bearer {vast_api_key}"
    }
    url = "https://console.vast.ai/api/v0/users/current"
    r = requests.get(url, headers=headers).json()
    return r

def get_all_offers(gpu_name=None):
    headers = {
        "Authorization": f"Bearer {vast_api_key}"
    }
    body = {
        "limit": 200000,
        # "external": {
        #     "eq": True,
        # },
        # "type": {
        #     "eq": "on-demand",
        # },
        "rentable": {
            "eq": True,
        },
    }
    if gpu_name is not None:
        body["gpu_name"] = {
            "eq": gpu_name
        }

    response = requests.post("https://console.vast.ai/api/v0/bundles/", headers=headers, json=body)
    response.raise_for_status()
    r = response.json()
    offers = pd.DataFrame(r["offers"])
    offers["bid_per_gpu"] = offers.min_bid / offers.num_gpus
    offers["bid_per_cpu"] = offers.min_bid / offers.cpu_cores_effective
    return offers

def create_instance(offer_id, bid, image, disk=5, env_vars={}):
    """Create an instance from the given offer. If bid is None than we use reserved
    pricing instead of spot pricing. Returns the instance id if successful, else None"""
    price = f"--bid_price {bid}" if bid is not None else ""

    env_arg = " ".join([f"-e {k}={v}" for k, v in env_vars.items()])

    cmd = f"vastai create instance {offer_id} --disk {disk} --image {image} {price} --login '-u {docker_username} -p {docker_password}' --env '{env_arg}' --args"
    # print(cmd)
    proc = subprocess.run(
        cmd,
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Get the instance id from stdout
    stdout = proc.stdout.decode("utf-8")
    # print(stdout)
    try:
        instance_id = int(stdout.split(" ")[-1].strip()[:-1])
    except ValueError:
        return None
    return instance_id

def get_all_instances():
    headers = {
        "Authorization": f"Bearer {vast_api_key}"
    }
    body = {}
    r = requests.get("https://console.vast.ai/api/v0/instances", headers=headers, json=body).json()
    return pd.DataFrame(r["instances"])

def change_bid_price(instance_id, bid):
    headers = {
        "Authorization": f"Bearer {vast_api_key}"
    }
    body = {
        "client_id": "me",
        "price": bid
    }
    url = f"https://console.vast.ai/api/v0/instance/bid_price/{instance_id}"
    r = requests.put(url, headers=headers, json=body)
    r.raise_for_status()

def destroy_instance(id):
    """Destroy the given instance"""
    cmd = f"vastai destroy instance {id}"
    subprocess.run(
        cmd,
        shell=True,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def normalize_gpu_name(name):
    return name.replace("NVIDIA", "").replace("GeForce", "").strip()

def get_running_machines():
    """Return a set of all the machines we're currently using"""
    df = get_all_instances()
    if len(df) == 0:
        return set()
    return set(df.machine_id)


def get_outbid_instances():
    instances = get_all_instances()
    outbid = (instances.actual_status == "loading") & (
        instances.intended_status == "stopped"
    )
    return instances[outbid]