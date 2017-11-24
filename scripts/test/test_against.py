"""test baselines in battle against"""

from search import do_task

task = [
    {
        "name": "against",
        "type": "single-search",
        "prefix": "python examples/train_against.py --train --save_every 100 --n_round 450",
        "arg_name": "--alg",
        "arg_value": ["a2c", "drqn", "dqn"]
    }
]

for item in task:
    do_task(item)
    print("%s done" % item['name'])

print("battle-against all done")
print("plot curve: python scripts/plot_many.py against-rec.log reward 0")

