/**
 * \file AgentType.h
 * \brief implementation of AgentType (mainly initialization)
 */

#ifndef MAGENT_GRIDWORLD_AGENTTYPE_H
#define MAGENT_GRIDWORLD_AGENTTYPE_H

#include <vector>

#include "grid_def.h"
#include "Range.h"

namespace magent {
namespace gridworld {

class AgentType {
public:
    AgentType(int n, std::string name, const char **keys, float *values, bool turn_mode);

    // user defined setting
    int width, length;
    float speed, hp;
    float view_radius, view_angle;
    float attack_radius, attack_angle;

    float hear_radius, speak_radius;

    int speak_ability;
    float damage, trace, eat_ability, step_recover, kill_supply, food_supply;
    bool attack_in_group;

    Reward step_reward, kill_reward, dead_penalty, attack_penalty;

    int view_x_offset, view_y_offset;
    int att_x_offset,  att_y_offset;
    int turn_x_offset, turn_y_offset;

    /** special for demo **/
    bool can_absorb;

    /***** system calculated setting *****/
    std::string name;
    int n_channel; // obstacle, group1, group_hp1, group2, group_hp2
    Range *view_range, *attack_range, *move_range;

    int move_base, turn_base, attack_base;
    std::vector<int> action_space;
};


} // namespace magent
} // namespace gridworld

#endif //MAGENT_GRIDWORLD_AGENTTYPE_H
