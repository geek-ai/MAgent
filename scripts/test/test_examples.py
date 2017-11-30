"""test examples"""

import os
import time

source = [
    "examples/train_tiger.py",
    "examples/train_pursuit.py",
    "examples/train_gather.py",
    "examples/train_battle.py",
    "examples/train_single.py",
    "examples/train_arrange.py",
    "examples/train_multi.py",
]


def do_cmd(cmd):
    tic = time.time()
    print(cmd)
    assert os.system(cmd) == 0
    return time.time() - tic


start = time.time()
for item in source:
    run_cmd = "python %s --train --n_round 1" % item
    do_cmd(run_cmd)

    change_cmd = "sed -i 's/tf_model/mx_model/g' %s" % item
    do_cmd(change_cmd)

    do_cmd(run_cmd)

print("test examples done", time.time() - start)
