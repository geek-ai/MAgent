# MAgent
MAgent is a platform aimed at many-agent reinforcement learning. 
Unlike previous research platforms on single or multi-agent reinforcement learning, 
MAgent focuses on supporting the tasks and the applications that require hundreds to millions of agents.
[see video](https://www.youtube.com/watch?v=HCSm0kVolqI)

# Requirement
MAgent currently support Linux and OS X running Python 2.7 or python 3
We make no assumptions about the structure of your agents.
You can write rule-based algorithm or use deep learning framework such as Tensorflow, MXNet, PyTorch.

# Install on linux

```bash
git clone git@bitbucket.org:geek-ai/magent.git

sudo apt-get install cmake libboost-system-dev libjsoncpp-dev libwebsocketpp-dev
bash build.sh
```

# Install on OSX
```bash
git clone git@bitbucket.org:geek-ai/magent.git

brew install cmake llvm
brew install boost
brew tap david-icracked/homebrew-websocketpp
brew install jsoncpp websocketpp
brew install argp-standalone

bash build.sh
```

## Examples
Examples need at least 16GB memory to run
The training time of following tasks is at most 1 day on GTX1080-Ti

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
