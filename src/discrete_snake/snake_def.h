/**
 * \file snake_def.h
 * \brief some global definition for discrete_snake
 */


#ifndef MAGNET_DISCRETE_SNAKE_SNAKEDEF_H
#define MAGNET_DISCRETE_SNAKE_SNAKEDEF_H

#include "../utility/utility.h"
#include "../Environment.h"

namespace magent {
namespace discrete_snake {

using ::magent::environment::Environment;
using ::magent::environment::GroupHandle;
using ::magent::utility::strequ;

struct Position {
    int x;
    int y;
};

typedef int PositionInteger;

typedef enum { ACT_RIGHT, ACT_DOWN, ACT_LEFT, ACT_UP, ACT_NOOP, ACT_NUM } Action;
typedef enum { RIGHT, DOWN, LEFT, UP, DIR_NUM } Direction;
enum { CHANNEL_WALL, CHANNEL_FOOD, CHANNEL_SELF, CHANNEL_OTHER, CHANNEL_ID, CHANNEL_NUM};
typedef float Reward;

class Agent;
class Food;
} // namespace discrete_snake
} // namespace magent

#endif //MAGNET_DISCRETE_SNAKE_SNAKEDEF_H
