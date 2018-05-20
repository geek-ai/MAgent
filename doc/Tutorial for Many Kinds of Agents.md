<center><b> A Tutotiral for Running Experiments with Many Kinds of Agents</b></center>

This tutorial will tell you how to modify the code to make it compatiable for running experiments with many kings of agents. In the following tutorial, we will take the *pursuit* environemtn as an example.
# Config Files
- Register all kinds of your agents through `cfg.regiser_agent_type`
- Add all your agents to the environment through `cfg.add_groups`
- Construct an symbol for each kind of your agent through `cfg.AgentSymbol`
- Finally, design the reward rules between all of your agents through `cfg.add_reward_rule`
 
You could refer to the [**Pursuit.py**](https://github.com/geek-ai/MAgent/blob/master/python/magent/builtin/config/pursuit.py) to easily finish these steps. Here is one thing you need to notice, when you do something wrong in modifying the config file like write a typo, the framework may just tell you that `there is no such an environment`. So carefully check you code and do not believe the error message.

# Training Code

There is mainly several parts you need to modify.

In the main part of the training code:

- Add all your agents' model into the list `models` 
- Add all your agents' name into the list `names`
- Add all the evaluation arguments into the list `eval_obs`

In the generating agents part of the code:

- Add all your kind of agents to the environemtn through `env.add_agents` (You may still get some incorrect error message in that step)

You could also refer to the [**train_pursuit.py**](https://github.com/geek-ai/MAgent/blob/master/examples/train_pursuit.py) to understand these instructions better. Another tip is that do not use multi-processing training code if the number of your agents is too large, since the gc of python is not quite well and it may lead you to the **OOM**.

Have fun with our framework (: