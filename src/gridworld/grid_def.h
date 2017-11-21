/**
 * \file grid_def.h
 * \brief some global definition for gridworld
 */

#ifndef MAGNET_GRIDWORLD_GRIDDEF_H
#define MAGNET_GRIDWORLD_GRIDDEF_H

#include "../Environment.h"
#include "../utility/utility.h"

namespace magent {
namespace gridworld {

typedef enum {EAST, SOUTH, WEST, NORTH, DIR_NUM} Direction;

typedef enum {
    OP_AND, OP_OR, OP_NOT,
    /***** split *****/
    OP_KILL, OP_AT, OP_IN, OP_COLLIDE, OP_ATTACK, OP_DIE,
    OP_IN_A_LINE, OP_ALIGN,
    OP_NULL,
} EventOp;


struct Position {
    int x, y;
};
typedef long long PositionInteger;

typedef float Reward;
typedef int   Action;

// some forward declaration
class Agent;
class AgentType;
class Group;

struct MoveAction;
struct TurnAction;
struct AttackAction;

// reward description
class AgentSymbol;
class RewardRule;
class EventNode;

using ::magent::environment::Environment;
using ::magent::environment::GroupHandle;
using ::magent::utility::strequ;
using ::magent::utility::NDPointer;

} // namespace gridworld
} // namespace magent


#endif //MAGNET_GRIDWORLD_GRIDDEF_H
