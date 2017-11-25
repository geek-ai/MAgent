"""test baselines in double attack"""

from search import do_task

task = [
    {
        "name": "tiger",
        "type": "single-search",
        "prefix": "python examples/train_tiger.py --train --n_round 250",
        "arg_name": "--alg",
        "arg_value": ["dqn", "a2c", "drqn"]
    }
]

for item in task:
    do_task(item)
    print("%s done" % item['name'])

print("tiger all done")
print("plot curve: python scripts/plot_many.py tiger-rec.out reward")
