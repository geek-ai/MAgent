# Environment
The basic environment is a large gridworld.

# Agents
agents are controlled by groups.

# Observation

# Action

# Run the first demo
```bash
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
python examples/api_demo.py
```

In this environment, blue agents are trained by Deep Q-Network, red agents act randomly.
Blue agents can only get rewards when two blue agents attack a red agents simultaneously.
So blue agents need to cooperate with each other.

# See the video
* Go to directory `build/render`
* Execute `./render`
* Open index.html in browser. A modal will be opened automatically once the frontend gets connected to the backend
* Type `config.json` and `video_1.txt` in the two input boxes.
* In the render, press arrow keys 'up', 'down', 'left', 'right' to move scope window. Press '<', '>' to zoom in or zoom out. For more help, press 'h'

# Next step
see 'train_tiger.py' to know how the above agents are trained.  
try other examples and have fun!
