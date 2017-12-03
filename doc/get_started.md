# Get started
This documentation will tell you the basic setting of MAgent.gridworld, and show how to run the first demo.

## Environment
The basic environment is a large gridworld.

## Agents
Agents are controlled by groups. Agents in a group share the same general attributes and control methods.

## Observation
There are two parts in observation, spacial local view and non-spacial feature. 


## Action
Actions are discrete actions. They can be move, turn, attack.

## Run the first demo
Run the following command in the root directory, do not cd to `examples/`
```bash
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
python examples/api_demo.py
```

In this environment, predators are pursuing preys. Predators can get rewards by attacking preys.
The predators and preys are trained by Deep Q-Network.
After training, predators learn to lock preys by cooperating with each other.  
See also 'train_pursuit.py' to know how the above agents are trained.

## See the video
* Go to directory `build/render`
* Execute `./render`
* Open index.html in browser. A modal will be opened automatically once the frontend gets connected to the backend
* Type `config.json` and `video_1.txt` in the two input boxes.
* In the render, press arrow keys 'up', 'down', 'left', 'right' to move scope window. Press '<', '>' to zoom in or zoom out. For more help, press 'h'

## Play the general-soldier game
In this section, Pygame are required.
```base
pip install pygame
```
Then you run the following script.
```bash
python examples/show_battle_game.py
```


## Next step
Try other examples and have fun!
