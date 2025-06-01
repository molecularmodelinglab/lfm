import dbm
import os
import sys
import traceback
from tqdm import tqdm

import warnings
from common.gs import GS_FS
from common.md_sim import BUCKET_FS, local_gs_path, reload_bucket_fs
from common.utils import CONFIG
from sqlalchemy.orm import Session

from common.db import PMFDatagen, exec_sql, get_db_gs_path, get_engine
from datagen.pmf_datagen import initialize_datagen

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import time
from common.vastai import (
    get_all_offers,
    get_all_instances,
    create_instance,
    destroy_instance,
    get_running_machines,
    get_user_info,
)

# these were obtained with an older version of the profiling code.
# todo: update these continuously
MD_GPU_COST_MULTIPLIERS = {
    "RTX 3060 Ti": 0.5431570448416221,
    "RTX 3070": 0.4799605542788581,
    "RTX A2000": 0.8391074151038524,
    "RTX 3060": 0.6307500393818518,
    "RTX 3090": 0.39533331300377544,
    "RTX A2000 12GB": 1.0,
    "RTX 3080": 0.4308703769559125,
    "RTX 4070": 0.37719183837972553,
    "RTX 4060 Ti": 0.4181575881612527,
    "RTX A5000": 0.4811363997576973,
    "RTX 3060 Laptop GPU": 0.6120140810367029,
    "RTX 3080 Ti": 0.4223834283248771,
    "RTX 6000 Ada Generation": 0.30958872331783455,
    "RTX 4070 Ti SUPER": 0.3307230685063414,
    "RTX A6000": 0.42823655805433314,
    "Tesla V100-FHHL-16GB": 0.5813743852541314,
    "A40": 0.4499522647666834,
}


class InstanceConductor:
    """Class for orchestrating the vast.ai instances. Inherit from this class to create a conductor for a specific task, eg datagen"""

    def __init__(
        self,
        image_name,
        max_cost_per_gpu,
        max_cost_per_cpu,
        gpu_cost_multipliers={},  # mapping between GPU name and effective cost relative to the "best" GPU. We want to pay less for bad GPUs
        env_vars={},  # environment variables to pass to the container
        image_tag="latest",
    ):

        self.image_name = image_name
        self.image = CONFIG.docker_prefic + "/" + image_name + ":" + image_tag
        self.max_cost_per_cpu = max_cost_per_cpu
        self.max_cost_per_gpu = max_cost_per_gpu
        self.env_vars = env_vars
        self.gpu_cost_multipliers = gpu_cost_multipliers

        # don't want to use the same machine twice after it failed
        # self.bad_machines = set()

        # self.instance2offer = {}
        # self.bad_offers = set()
        self.instance2offer = dbm.open("output/instance2offer.dbm", "c")
        self.bad_offers = dbm.open("output/bad_offers.dbm", "c")

        self.log_fname = f"output/{self.image_name}_log.csv"
        if os.path.exists(self.log_fname):
            self.log = pd.read_csv(self.log_fname)
        else:
            self.log = pd.DataFrame(
                columns=["vast_credit", "n_instances"] + self.get_extra_log_columns()
            )

    def modernize_bad_offers(self):
        """ Sets the bad_offers unix time to now for all bad offers """
        for key in self.bad_offers.keys():
            offer_id = int(key.decode())
            num_bad, _ = self.get_current_offer_badness(offer_id)
            unix_time = int(time.time())
            self.bad_offers[key] = f"{num_bad},{unix_time}".encode()

    def instance_to_offer(self, instance_id):
        """ Get the offer id for a given instance id. May raise KeyError """
        return int(self.instance2offer[str(instance_id).encode()].decode())

    def get_current_offer_badness(self, offer_id):
        """ Returns the current badness of the offer as (num_bad, unix_time). Default (0,0) """
        key = str(offer_id).encode()
        num_str, time_str = self.bad_offers.get(key, b"0,0").decode().split(",")
        return int(num_str), int(time_str)

    def increment_bad_offer(self, offer_id):
        """ Sets the current offer badness to be (cur_num + 1, unix_time) """
        key = str(offer_id).encode()
        cur_num, _ = self.get_current_offer_badness(offer_id)
        unix_time = int(time.time())
        self.bad_offers[key] = f"{cur_num + 1},{unix_time}".encode()

    def decrement_bad_offer(self, offer_id):
        """ Decrements the badness of the offer by 1. Keeps the time the same """
        key = str(offer_id).encode()
        cur_num, unix_time = self.get_current_offer_badness(offer_id)
        if cur_num == 0:
            # nothing to do
            return
        elif cur_num == 1:
            # remove the offer from the bad_offers
            del self.bad_offers[key]
        elif cur_num > 1:
            self.bad_offers[key] = f"{cur_num - 1},{unix_time}".encode()

    def set_bad_instance(self, instance_id):
        """ Increments the badness of the instance offer """
        try:
            offer_id = self.instance_to_offer(instance_id)
        except KeyError:
            print(f"Instance {instance_id} not in instance2offer, deleting")
            destroy_instance(instance_id)
            return
        
        self.increment_bad_offer(offer_id)

    def get_bad_offers(self, backoff_time=120*60):
        """ Returns all offers that have been bad for less than backoff_time*(2^(num_bad - 1)) seconds """
        bad_offers = []
        for key in self.bad_offers.keys():
            offer_id = int(key.decode())
            num_bad, unix_time = self.get_current_offer_badness(offer_id)
            assert num_bad >= 0

            bad_time = backoff_time * (2 ** (num_bad - 1))
            if time.time() - unix_time < bad_time:
                bad_offers.append(int(offer_id))
            # else:
            #     time_delta_min = (time.time() - unix_time) / 60
            #     print(f"Giving offer {offer_id} another chance after {time_delta_min} minutes")
        return set(bad_offers)

    def reduce_all_offer_badness(self, recovery_time=240*60):
        """ Reduces the badness of all offers by 1, if they have been bad for more than recovery_time*2^(num_bad - 1) seconds """
        for key in self.bad_offers.keys():
            offer_id = int(key.decode())
            num_bad, unix_time = self.get_current_offer_badness(offer_id)
            bad_time = recovery_time * (2 ** (num_bad - 1))
            if time.time() - unix_time > bad_time:
                time_delta_min = (time.time() - unix_time) / 60
                print(f"Reducing badness of offer {offer_id} after {time_delta_min} minutes")
                self.decrement_bad_offer(offer_id)

    def __del__(self):
        self.instance2offer.close()
        self.bad_offers.close()

    def get_extra_log_columns(self):
        """Override this to add additional columns to the log"""
        return []

    def get_offers(self, exclude_running=True):
        """Get all the offers for machines that we're not currently using
        and that aren't in bad_machines"""
        # excluded = self.bad_machines
        # if exclude_running:
        #     excluded.update(get_running_machines())
        excluded = self.get_bad_offers()
        if exclude_running:
            current_instances = get_all_instances()
            current_offers_str = { self.instance2offer.get(str(instance.id), None) for i, instance in current_instances.iterrows() }
            excluded.update({int(offer) for offer in current_offers_str if offer is not None})

        offers = get_all_offers()

        # divide cost by multipliers. We want to pay less for bad GPUs
        cost_multipliers = [
            self.gpu_cost_multipliers.get(name, 1) for name in offers.gpu_name
        ]
        offers.bid_per_gpu /= cost_multipliers

        # only consider offers with N CU >= N GPU

        offers = offers[offers.cpu_cores_effective >= offers.num_gpus]

        # return offers[~offers.machine_id.isin(excluded)]
        return offers[~offers.id.isin(excluded)]

    def create_instances(self, offers):
        """Create all instances for the sufficiently cheap offers"""

        mask = np.ones(len(offers), dtype=bool)
        if self.max_cost_per_cpu is not None:
            mask &= offers.bid_per_cpu <= self.max_cost_per_cpu
        if self.max_cost_per_gpu is not None:
            mask &= offers.bid_per_gpu <= self.max_cost_per_gpu

        to_create = offers[mask]
        for i, offer in tqdm(to_create.iterrows(), total=len(to_create)):
            instance_id = create_instance(
                offer.id, offer.min_bid, self.image, env_vars=self.env_vars
            )
            if instance_id is not None:
                # self.instance2offer[instance_id] = offer.id
                self.instance2offer[str(instance_id)] = str(offer.id)

    def destroy_instance(self, instance_id):
        """Destroy the given instance"""
        destroy_instance(instance_id)

    def destroy_nonrunning(self, max_age_minutes=20):
        """Destroys outbid and exited instances. Also destroys loading instances after 20 minutes and adds them to bad_machines"""
        instances = get_all_instances()
        nonrunning_mask = instances.actual_status != "running"

        nonrunning = instances[nonrunning_mask]
        for i, instance in nonrunning.iterrows():

            if (
                instance.intended_status == "stopped"
                or instance.actual_status == "exited"
            ):
                # destroy but don't add to bad_machines
                destroy_instance(instance.id)
            else:
                age = time.time() - instance.start_date
                age_minutes = age / 60
                if age_minutes > max_age_minutes:
                    print(
                        f"Destroying instance {instance.id} after {age_minutes:.2f} minutes"
                    )
                    destroy_instance(instance.id)

    def destroy_incompetent(self):
        """Destroys instances that are not completing tasks fast enough. Override this for specific use cases"""
        pass

    def get_extra_log_info(self):
        """Override to return additional logging information"""
        return {}

    def update_log(self):
        """Updates the log with whatever attributes we return in get_extra_log_info()"""
        n_instances = len(get_all_instances())
        vast_credit = get_user_info()["credit"]
        row = {
            "time": time.time(),
            "vast_credit": vast_credit,
            "n_instances": n_instances,
        }
        row.update(self.get_extra_log_info())
        for key, value in row.items():
            print(f"{key}: {value}")

        to_append = pd.DataFrame([row])
        self.log = pd.concat([self.log, to_append])
        self.log.to_csv(self.log_fname, index=False)

    def reset(self):
        """Kills running instances"""
        instances = get_all_instances()
        for i, instance in tqdm(instances.iterrows(), total=len(instances)):
            destroy_instance(instance.id)
        # remove all instances from instance2offer
        self.instance2offer.clear()

    def run_once(self):
        print("==== Running loop ====")

        print("Getting offers")
        offers = self.get_offers()

        print("Creating instances")
        self.create_instances(offers)

        time.sleep(60 * 1)

        print("Destroying instances")

        self.destroy_incompetent()
        self.destroy_nonrunning()

        print("Recalculating bad offers")
        self.reduce_all_offer_badness()

        time.sleep(60 * 1)

    def run(self):
        """Run the conductor loop"""

        while True:
            try:

                self.update_log()
                self.run_once()

            except KeyboardInterrupt:
                raise
            except:
                print("Error in main loop, retrying")
                traceback.print_exc()


class PMFDatagenConductor(InstanceConductor):
    """Conductor for PMF datagen"""

    def __init__(self, target_name, datagen_name, sim_name, max_cost_per_gpu):
        self.datagen_id = initialize_datagen(target_name, datagen_name, sim_name)
        self.datagen_path = get_db_gs_path("pmf_datagens", self.datagen_id)
        self.engine = get_engine()

        env_vars = {"DATAGEN_ID": self.datagen_id}

        super().__init__(
            "pmf_datagen",
            max_cost_per_gpu,
            max_cost_per_cpu=None,
            gpu_cost_multipliers=MD_GPU_COST_MULTIPLIERS,
            env_vars=env_vars,
        )

    def destroy_incompetent(
        self, max_age_hours=1, max_failures=30, max_task_minutes=35, check_hours=1
    ):
        """Destroys instances that haven't made any datapoints after 1 hour and adds them to bad_machines"""

        instances = get_all_instances()

        if check_hours is not None:
            time_clause = f" AND start_time > NOW() - INTERVAL '{check_hours} HOURS'"
        else:
            time_clause = ""

        logs = exec_sql(
            self.engine,
            f"SELECT error_type, vastai_instance_id, n_gpu, total_time, success from pmf_datagen_results WHERE datagen_id={self.datagen_id} {time_clause}",
        )
        instance_ids = logs.vastai_instance_id.str.split(".").apply(
            lambda x: int(x[1]) if isinstance(x, list) else None
        )

        for instance in instances.itertuples():
            instance_logs = logs[instance_ids == instance.id]

            # if any ConnectionError has occured, kill it but just restart it
            # These tend to happen because of ZINC API rate limiting
            if "ConnectionError" in instance_logs.error_type.unique():
                print(f"Destroying instance {instance.id} after ConnectionError")
                destroy_instance(instance.id)
                continue

            # first kill it if it keeps erroring out, even if it hasn't been an hour
            if instance_logs.success.sum() == 0 and len(instance_logs) > max_failures:
                print(
                    f"Destroying instance {instance.id} after {max_failures} failures"
                )
                destroy_instance(instance.id)
                self.set_bad_instance(instance.id)
                continue

            # kill it if any FatalError has occurred
            if "FatalError" in instance_logs.error_type.unique():
                print(f"Destroying instance {instance.id} after FatalError")
                destroy_instance(instance.id)
                self.set_bad_instance(instance.id)
                continue

            # now kill it if any task from this instance has taken too long
            if instance_logs.total_time.max().total_seconds() / 60 > max_task_minutes:
                print(f"Destroying instance {instance.id} after task took too long")
                destroy_instance(instance.id)
                self.set_bad_instance(instance.id)
                continue

            if instance.actual_status != "running":
                continue

            age = time.time() - instance.start_date
            age_hours = age / 3600
            if age_hours > max_age_hours:
                # each GPU should have made at least one datapoint by now
                if (
                    len(instance_logs) == 0
                    or instance_logs.success.sum() < instance_logs.n_gpu.iloc[0]
                ):
                    print(
                        f"Destroying instance {instance.id} after {age_hours:.2f} hours"
                    )
                    destroy_instance(instance.id)
                    # self.bad_machines.add(instance.machine_id)
                    # self.bad_offers.add(self.instance2offer[instance.id])
                    self.set_bad_instance(instance.id)

    def get_num_datapoints(self):
        try:
            return (
                len(
                    GS_FS.ls(
                        self.datagen_path.replace("gs://", ""), refresh=True
                    )
                )
            )
        except FileNotFoundError:
            return 0

    def get_extra_log_columns(self):
        return ["n_datapoints"]

    def get_extra_log_info(self):
        return {"n_datapoints": self.get_num_datapoints()}


if __name__ == "__main__":
    action = sys.argv[1]
    target_name = sys.argv[2]
    datagen_name = sys.argv[3]
    sim_name = sys.argv[4]
    if sim_name == "None":
        sim_name = None

    max_cost_per_gpu = float(sys.argv[5])
    conductor = PMFDatagenConductor(target_name, datagen_name, sim_name, max_cost_per_gpu=max_cost_per_gpu)
    
    # if we haven't run in a while
    if "--modernize" in sys.argv:
        conductor.modernize_bad_offers()
    
    match action:
        case "run":
            conductor.run()
        case "stop":
            conductor.reset()
        case _:
            raise ValueError(f"Unknown action {action}")