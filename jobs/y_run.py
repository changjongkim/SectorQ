#!/usr/bin/env python3
"""Runner of the review experiments (queue_y, logs/y): jobs/g_run.py with its queue and log directories moved.

  python3 jobs/y_run.py --job <allocation id> --hosts <host[,host...]>
  python3 jobs/y_run.py --count
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g_run                                    # noqa: E402

g_run.Q = os.path.join(g_run.ROOT, "queue_y")
g_run.LOGS = os.path.join(g_run.ROOT, "logs", "y")
_plain = g_run.run_engine


def run_engine(item, job, hosts):
    record = _plain(item, job, hosts)
    record["log"] = f"logs/y/{item['id']}.out"     # g_run writes the path of campaign g into the record
    return record


g_run.run_engine = run_engine

if __name__ == "__main__":
    os.makedirs(g_run.LOGS, exist_ok=True)
    g_run.main()
