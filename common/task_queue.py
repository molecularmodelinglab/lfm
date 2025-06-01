import sys
import time
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from traceback import print_exc
from common.db import Task, TaskStatus, exec_sql

class TaskFunction:
    """Wrapper for a function we want to execute remotely"""

    all_task_funcs = {}

    def __init__(self, f, max_runtime_hours, retry_failed, prefetch):
        self.f = f
        self.max_runtime_hours = max_runtime_hours
        self.retry_failed = retry_failed
        self.prefetch = prefetch
        TaskFunction.all_task_funcs[f.__name__] = self

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def run_task(self, engine, task_id, args, kwargs):
        """Run the task and update the DB"""

        remove_task = True

        try:
            self.f(*args, **kwargs)
            print(f"Task {self.f.__name__} {task_id} completed successfully")
        except KeyboardInterrupt:
            raise
        except:
            print(f"Error running task {self.f.__name__} {task_id}:")
            print_exc()

            if self.retry_failed:
                status = TaskStatus.FAILED.value

                exec_sql(
                    engine, f"UPDATE tasks SET status = {status} WHERE id = {task_id}", fetch=False
                )
                remove_task = False

        if remove_task:
            exec_sql(engine, f"DELETE FROM tasks WHERE id = {task_id}", fetch=False)
            print(f"Task {self.f.__name__} {task_id} removed from queue")

    def delay(self, engine, *args, **kwargs):
        """ Queue the task to the DB"""
        with Session(engine) as sess:
            task = Task(
                name=self.f.__name__,
                args=(args, kwargs),
                status=TaskStatus.PENDING,
            )
            sess.add(task)
            sess.commit()
            task_id = task.id

        return task_id
    
    def delay_bulk(self, engine, args_list, kwargs_list=None):
        """ Bulk queue tasks to the DB. Should be more efficient """
        if kwargs_list is None:
            kwargs_list = [{}] * len(args_list)

        assert len(args_list) == len(kwargs_list)
        with Session(engine) as sess:
            tasks = []
            for args, kwargs in zip(args_list, kwargs_list):
                task = Task(
                    name=self.f.__name__,
                    args=(args, kwargs),
                    status=TaskStatus.PENDING,
                )
                tasks.append(task)
            sess.add_all(tasks)
            sess.commit()
            # task_ids = [task.id for task in tasks]

        # lol takes too long to return all the task ids
        # return task_ids

    def clear_queue(self, engine):
        """ Clear the task queue """
        exec_sql(engine, f"DELETE FROM tasks WHERE name = '{self.f.__name__}'", fetch=False)

def task(max_runtime_hours=12, retry_failed=False, prefetch=10):
    """Decorator for a function we want to execute remotely"""
    def decorator(f):
        return TaskFunction(f, max_runtime_hours, retry_failed, prefetch)

    return decorator

def get_task_select_query(task_name, task_func):
    """ Returns the SQL query to select tasks to run """

    from_query = f"""
        SELECT id FROM tasks
        WHERE name = '{task_name}'
        AND status != 'RUNNING' OR expires_at < now()
        LIMIT {task_func.prefetch}
        FOR UPDATE SKIP LOCKED
    """

    if task_func.max_runtime_hours is None and not task_func.retry_failed:
        # in this case, just immediately remove the task
        query = f"""
        DELETE FROM tasks
        WHERE tasks.id IN (
            {from_query}
        )
        RETURNING id, args
        """

    else:
        if task_func.max_runtime_hours is None:
            exp_clause = ""
        else:
            exp_clause = f", expires_at = now() + interval '{task_func.max_runtime_hours} hours'"

        query = f"""
        UPDATE tasks
        SET status = 'RUNNING'{exp_clause}
        FROM (
            {from_query}
        ) AS running_tasks
        WHERE tasks.id = running_tasks.id
        RETURNING tasks.id, tasks.args
        """

    return query

def task_loop(engine, task_name, wait_time_seconds=10):
    """ 
    Loop that checks for tasks to run and runs them
    """

    print(f"Starting task loop for {task_name}...")
    task_func = TaskFunction.all_task_funcs[task_name]
    query = get_task_select_query(task_name, task_func)


    while True:
        try:

            task_df = exec_sql(engine, query, transaction=False)
            # if we get no tasks, sleep and try again
            if len(task_df) == 0:
                time.sleep(wait_time_seconds)
                continue

            for task_id, (args, kwargs) in task_df.itertuples(index=False):
                task_func.run_task(
                    engine,
                    task_id,
                    args,
                    kwargs,
                )
                sys.stdout.flush()

        except KeyboardInterrupt:
            raise
            print("Error in task loop:")
            print_exc()
            time.sleep(wait_time_seconds)

        sys.stdout.flush()


