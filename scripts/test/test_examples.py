"""test examples"""

import os
import time

tasks = [
    "python examples/train_tiger.py   --train --n_round 1",
    "python examples/train_pursuit.py --train --n_round 1",
    "python examples/train_gather.py  --train --n_round 1",
    "python examples/train_battle.py  --train --n_round 1",
    "python examples/train_single.py  --train --n_round 1",
    "python examples/train_arrange.py --train --n_round 1",
    "python examples/train_multi.py   --train --n_round 1",
]

start = time.time()
for item in tasks:
    cmd = item
    print(cmd)
    tic = time.time()
    ret = os.system(cmd)
    print(cmd, time.time() - tic)
    assert ret == 0

print("test examples done", time.time() - start)
