# Get started
This tutorial will tell you the basic setting of the gridworld in MAgent, and show how to run the first demo.

## Environment
The basic environment is a large gridworld. Agents and walls can be added in the world.

## Agents
Agents are controlled by groups. Agents in a group share the same general attributes and control method.
The attributes of an agent can be width, height, speed, hp, etc. An AgentType is registered as follows.
```python
predator = register_agent_type(
    "predator",
    {
        'width': 2, 'length': 2, 'hp': 1, 'speed': 1,
        'view_range': CircleRange(5), 'attack_range': CircleRange(2),
        'attack_penalty': -0.2
    })
```

## Observation
There are two parts in observation, spacial local view and non-spacial feature (see figure below).
Spatial view consists of several channels. They will be masked by a circle or a sector.
Non-spatial feature includes ID embedding, last action, last reward and normalized position.
ID embedding is the binary representation of agent's unique ID.

<img src="../data/figure/observation_space.png" width="350">

## Action
Actions are discrete actions. They can be move, attack and turn.
In the figure below, move range and attack range are also circular range (chunked to fit grids).

<img src="../data/figure/action_space.png" width="300">

## Reward
Reward can be defined by constant attributes of agent type or by event trigger.
See [python/magent/builtin/config](../python/magent/builtin/config/) for more examples.
Of course, you can also write your own reward rules in the control logic in python code.

## Run the first demo
Run the following command in the root directory, do not cd to `examples/`
```bash
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
python examples/api_demo.py
```

In this environment, predators are pursuing preys. Predators can get rewards by attacking preys.
The predators and preys are trained by Deep Q-Network.
After training, predators learn to lock preys by cooperating with each other.  
See also [train_pursuit.py](../examples/train_pursuit.py) to know how the above agents are trained.

## Watch video
* Go to directory `build/render`
* Execute `./render`
* Open index.html in browser. A modal will be opened once the frontend gets connected to the backend
* Type `config.json` and `video_1.txt` in the two input boxes.
* In the render, press arrow keys 'up', 'down', 'left', 'right' to move scope window. Press '<', '>' to zoom in or zoom out. For more help, press 'h'

## Play general-soldier game
In this section, Pygame are required.  
**Note for OSX user**: Unluckily, there is something wrong with Pygame on OSX, which makes this game very slow. You can skip this game if you are on OSX.

```base
pip install pygame
python examples/show_battle_game.py
```
In this game, you will act as a general and dispatch your soilders.
You have 10 chances to place your soilders in the map.
Then the soilders will act according to their deep q-networks.
You goal is to find best places to place your soilders and let them to eliminate the enemies.

## Next step
Try other examples and have fun!
