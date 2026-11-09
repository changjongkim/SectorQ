#!/usr/bin/env python3
"""The runner of campaign g (queue_g, logs/g) with the watch on stalled engine runs of jobs/s_run_w.py.

  python3 jobs/gw_run.py --job <job id> --hosts <host[,host...]>
  python3 jobs/gw_run.py --count

jobs/s_run_w.py wraps the runner of the scale campaign (queue_s). Importing it points g_run at queue_s and puts
s_run's log rewrite (logs/s) under the watch; both are set back here, so the watch runs on campaign g's own runs.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g_run                                    # noqa: E402

_plain = g_run.run_engine
import s_run_w                                  # noqa: E402  (sets g_run.Q/LOGS to the scale campaign)

g_run.Q = os.path.join(g_run.ROOT, "queue_g")
g_run.LOGS = os.path.join(g_run.ROOT, "logs", "g")
s_run_w._run_engine = _plain                    # the watch calls the plain runner (logs/g), not s_run's (logs/s)
g_run.run_engine = s_run_w.run_engine

if __name__ == "__main__":
    g_run.main()
