from search import do_task

task = [
    {
        "name": "against",
        "type": "single-search",
        "prefix": "python examples/train_against.py --train --n_round 600",
        "arg_name": "--alg",
        "arg_value": ["dqn", "a2c", "drqn"]
    }
]

for item in task:
    do_task(item)
    print("%s done" % item['name'])

print("battle-against all done")
print("plot curve: python scripts/plot_many.py against-rec.log reward 0")

