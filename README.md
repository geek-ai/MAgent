<h1><img src="data/figure/logo.png" width="200"></h1>

MAgent is a research platform for many-agent reinforcement learning.
Unlike previous research platforms that focus on reinforcement learning research with a single agent or only few agents, 
MAgent aims at supporting reinforcement learning research that scales up from hundreds to millions of agents.

- AAAI 2018 demo paper: [MAgent: A Many-Agent Reinforcement Learning Platform for Artificial Collective Intelligence](https://arxiv.org/abs/1712.00600)
- Watch [our demo video](https://www.youtube.com/watch?v=HCSm0kVolqI) for some interesting show cases.
- Here are two immediate demo for the battle case.

<img src="https://kipsora.github.io/resources/magent-graph-1.gif" width="200"><img src="https://kipsora.github.io/resources/magent-graph-2.gif" width="200">

## Requirement
MAgent supports Linux and OS X running Python 2.7 or python 3.
We make no assumptions about the structure of your agents.
You can write rule-based algorithms or use deep learning frameworks.

## Install on Linux

```bash
git clone git@github.com:geek-ai/MAgent.git
cd MAgent

sudo apt-get install cmake libboost-system-dev libjsoncpp-dev libwebsocketpp-dev

bash build.sh
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
```

## Install on OSX
**Note: There is an issue with homebrew for installing websocketpp, please refer to [#17](https://github.com/geek-ai/MAgent/issues/17)**
```bash
git clone git@github.com:geek-ai/MAgent.git
cd MAgent

brew install cmake llvm boost
brew install jsoncpp argp-standalone
brew tap david-icracked/homebrew-websocketpp
brew install --HEAD david-icracked/websocketpp/websocketpp

bash build.sh
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
```

## Docs
[Get started](/doc/get_started.md)


## Examples
The training time of following tasks is about 1 day on a GTX1080-Ti card.
If out-of-memory errors occur, you can tune infer_batch_size smaller in models.

**Note** : You should run following examples in the root directory of this repo. Do not cd to `examples/`.

### Train
Three examples shown in the above video.
Video files will be saved every 10 rounds. You can use render to watch them.

* **pursuit**

	```
	python examples/train_pursuit.py --train
	```

* **gathering**

	```
	python examples/train_gather.py --train
	```

* **battle**

	```
	python examples/train_battle.py --train
	```
### Play
An interactive game to play with battle agents. You will act as a general and dispatch your soldiers.

* **battle game**
    ```
    python examples/show_battle_game.py
    ```

## Baseline Algorithms
The baseline algorithms parameter-sharing DQN, DRQN, a2c are implemented in Tensorflow and MXNet.
DQN performs best in our large number sharing and gridworld settings.

## Acknowledgement
Many thanks to [Tianqi Chen](https://tqchen.github.io/) for the helpful suggestions.
