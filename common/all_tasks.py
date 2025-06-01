# import all celery tasks except yank
# and now our own task code -- currently concurrent with celery but
# I'm phasing it out
from common.baselines import *
from common.md_sim import *
from common.pmf_screen import *
from common.model_scoring import *
from common.task_queue import task_loop
from pmf_net.train import *
from datagen.collate import *
from common.vina import *
from analysis.true_pmf import *

# from ncmc.cmc_tasks import *

if __name__ == "__main__":
    import sys

    task_name = sys.argv[1]
    engine = get_engine()
    task_loop(engine, task_name)