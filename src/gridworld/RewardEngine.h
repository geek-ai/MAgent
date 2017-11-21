/**
 * \file reward_description.h
 * \brief Data structure for reward description
 */

#ifndef MAGNET_GRIDWORLD_REWARD_DESCRIPTION_H
#define MAGNET_GRIDWORLD_REWARD_DESCRIPTION_H

#include <vector>
#include <set>
#include <map>
#include "grid_def.h"

namespace magent {
namespace gridworld {

class AgentSymbol {
public:
    int group;
    int index;           // -1 for any, -2 for all
    void *entity;

    bool is_all() {
        return index == -2;
    }

    bool is_any() {
        return index == -1;
    }

    bool bind_with_check(void *entity);
};

class EventNode {
public:
    EventOp op;
    std::vector<AgentSymbol*> symbol_input;
    std::vector<EventNode*>   node_input;
    std::vector<int>          int_input;

    std::set<AgentSymbol*> related_symbols;
    std::map<AgentSymbol*, AgentSymbol*> infer_map;

    std::vector<int> raw_parameter;  // serialized parameter from python end
};

class RewardRule {
public:
    std::vector<AgentSymbol*> input_symbols;
    std::vector<AgentSymbol*> infer_obj;
    EventNode *on;

    std::vector<AgentSymbol*> receivers;
    std::vector<float> values;
    bool is_terminal;
    bool auto_value;

    std::vector<int> raw_parameter; // serialized parameter from python end

    bool trigger;
};

} // namespace gridworld
} // namespace magent

#endif //MAGNET_GRIDWORLD_REWARD_DESCRIPTION_H
