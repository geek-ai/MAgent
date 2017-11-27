MAgent
==============================================

[![stability-experimental](https://img.shields.io/badge/stability-experimental-orange.svg)](https://github.com/emersion/stability-badges#experimental)
[![Build Status](http://oj.kipsora.com:8080/buildStatus/icon?job=magent)]()


MAgent is a platform aimed at many-agent reinforcement learning.
Unlike previous research platforms on single or multi-agent reinforcement learning, 
MAgent focuses on supporting the tasks and the applications that require hundreds to millions of agents.
[see video](https://www.youtube.com/watch?v=HCSm0kVolqI)

## Requirement
MAgent currently support Linux and OS X running Python 2.7 or python 3.
We make no assumptions about the structure of your agents.
You can write rule-based algorithms or use deep learning frameworks such as Tensorflow, MXNet, PyTorch.

## Install on Linux

```bash
git clone git@bitbucket.org:geek-ai/magent.git
cd MAgent

sudo apt-get install cmake libboost-system-dev libjsoncpp-dev libwebsocketpp-dev

bash build.sh
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
```

## Install on OSX
```bash
git clone git@bitbucket.org:geek-ai/magent.git
cd MAgent

brew install cmake llvm
brew install boost
brew tap david-icracked/homebrew-websocketpp
brew install jsoncpp websocketpp
brew install argp-standalone

bash build.sh
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
```

## Docs
[Get started](/doc/get_started.md)


## Examples
The training time of following tasks is at most 1 day on a GTX1080-Ti card.
If you meet out-of-memory error, you can tune the map_size smaller or tune infer_batch_size smaller in models.

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

Video files will be saved every 10 rounds. You can use render to see them.

## Baseline Algorithm
Baseline algorithm is implemented both in Tensorflow and MXNet.
