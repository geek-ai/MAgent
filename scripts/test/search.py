"""do search task"""

import os
import sys
import argparse
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def do_task(task_item):
    recorder = open(task_item["name"] + "-rec.out", "w")
    for value in task_item["arg_value"]:
        tmp_name = task_item["name"] + "-" + value
        cmd = " ".join([task_item["prefix"], task_item["arg_name"], value])
        # cmd = "time " + cmd
        cmd += " --name " + tmp_name
        cmd = cmd + " >> " + tmp_name + ".out"
        print("%s : %s" % (tmp_name, cmd))
        start = time.time()
        os.system(cmd)
        use_time = time.time() - start
        recorder.write("log_file: %s\t time: %.2f\n" % (tmp_name + ".log", use_time))
    recorder.close()

