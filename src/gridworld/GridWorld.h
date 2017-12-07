/**
 * \file GridWorld.h
 * \brief core game engine of the gridworld
 */

#ifndef MAGNET_GRIDWORLD_GRIDWORLD_H
#define MAGNET_GRIDWORLD_GRIDWORLD_H

#include <vector>
#include <map>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "../Environment.h"
#include "grid_def.h"
#include "Map.h"
#include "Range.h"
#include "AgentType.h"
#include "RenderGenerator.h"
#include "RewardEngine.h"

namespace magent {
namespace gridworld {


// the statistical recorder
struct StatRecorder {
    int both_attack;

    void reset() {
        both_attack = 0;
    }
};


// the main engine
class GridWorld: public Environment {
public:
    GridWorld();
    ~GridWorld() override;

    // game
    void reset() override;
    void set_config(const char *key, void *p_value) override;

    // run step
    void get_observation(GroupHandle group, float **linear_buffers) override;
    void set_action(GroupHandle group, const int *actions) override;
    void step(int *done) override;
    void get_reward(GroupHandle group, float *buffer) override;

    // info getter
    void get_info(GroupHandle group, const char *name, void *buffer) override;

    // render
    void render() override;

    // special run step
    void clear_dead();
    void set_goal(GroupHandle group, const char *method, const int *linear_buffer);

    // agent
    void register_agent_type(const char *name, int n, const char **keys, float *values);
    void new_group(const char *agent_name, GroupHandle *group);
    void add_agents(GroupHandle group, int n, const char *method,
                    const int *pos_x, const int *pos_y, const int *pos_dir);

    // reward description
    void define_agent_symbol(int no, int group, int index);
    void define_event_node(int no, int op, int *inputs, int n_inputs);
    void add_reward_rule(int on, int *receivers, float *values, int n_receiver,
                         bool is_terminal, bool auto_value);

private:
    // reward description
    void init_reward_description();
    void calc_reward();
    void calc_rule(std::vector<AgentSymbol *> &input_symbols, std::vector<AgentSymbol *> &infer_obj,
                   RewardRule &rule, int now);
    bool calc_event_node(EventNode *node, RewardRule &rule);
    void collect_related_symbol(EventNode &node);

    // utility
    // to make channel layout in observation symmetric to every group
    std::vector<int> make_channel_trans(
            GroupHandle group, int base, int n_channel, int n_group);
    int group2channel(GroupHandle group);
    int get_feature_size(GroupHandle group);

    // game config
    int width, height;
    bool food_mode;      // default = False
    bool turn_mode;      // default = False
    bool minimap_mode;   // default = False
    bool goal_mode;      // default = False
    bool large_map_mode; // default = False
    bool mean_mode;      
    int embedding_size;  // default = 0

    // game states : map, agent and group
    Map map;
    std::map<std::string, AgentType> agent_types;
    std::vector<Group> groups;
    std::default_random_engine random_engine;

    // reward description
    std::vector<AgentSymbol> agent_symbols;
    std::vector<EventNode>   event_nodes;
    std::vector<RewardRule>  reward_rules;
    bool reward_des_initialized;

    // action buffer
    std::vector<AttackAction> attack_buffer;
    // split the events to small regions and boundary for parallel
    int NUM_SEP_BUFFER;
    std::vector<MoveAction> *move_buffers, move_buffer_bound;
    std::vector<TurnAction> *turn_buffers, turn_buffer_bound;

    // render
    RenderGenerator render_generator;
    int id_counter;
    bool first_render;

    // statistic recorder
    StatRecorder stat_recorder;
    int *counter_x, *counter_y;
};


class Agent {
public:
    Agent(AgentType &type, int id, GroupHandle group) : dead(false), absorbed(false), group(group),
                                                        next_reward(0),
                                                        type(type),
                                                        last_op(OP_NULL), op_obj(nullptr), index(0) {
        this->id = id;
        dir = Direction(rand() % 4);
        hp = type.hp;
        last_action = static_cast<Action>(type.action_space.size()); // dangerous here !
        next_reward = 0;

        init_reward();
    }

    Position &get_pos()             { return pos; }
    const Position &get_pos() const { return pos; }
    void set_pos(Position pos) { this->pos = pos; }

    Direction get_dir() const   { return dir; }
    void set_dir(Direction dir) { this->dir = dir; }

    AgentType &get_type()             { return type; }
    const AgentType &get_type() const { return type; }

    int get_id() const            { return id; }
    void get_embedding(float *buf, int size) {
        // embedding are binary form of id
        if (embedding.empty()) {
            int t = id;
            for (int i = 0; i < size; i++, t >>= 1) {
                embedding.push_back((float)(t & 1));
            }
        }
        memcpy(buf, &embedding[0], sizeof(float) * size);
    }

    void init_reward() {
        last_reward = next_reward;
        last_op = OP_NULL;
        next_reward = type.step_reward;
        op_obj = nullptr;
        be_involved = false;
    }
    Reward get_reward()         { return next_reward; }
    Reward get_last_reward()    { return last_reward; }
    void add_reward(Reward add) { next_reward += add; }

    void set_involved(bool value) { be_involved = value; }
    bool get_involved() { return be_involved; }

    void set_action(Action act) { last_action = act; }
    Action get_action()         { return last_action; }

    void add_hp(float add) { hp =  std::min(type.hp, hp + add); }
    float get_hp() const   { return hp; }
    void set_hp(float value) { hp = value; }

    bool is_dead() const { return dead; }
    void set_dead(bool value) { dead = value; }
    bool is_absorbed() const { return absorbed; }
    void set_absorbed(bool value) { absorbed = value; }

    bool starve() {
        if (type.step_recover > 0) {
            add_hp(type.step_recover);
        }
        else
            be_attack(-type.step_recover);
        return dead;
    }

    void be_attack(float damage) {
        hp -= damage;
        if (hp < 0.0) {
            dead = true;
            next_reward = type.dead_penalty;
        }
    }

    GroupHandle get_group() const { return group; }
    int get_index() const { return index; }
    void set_index(int i) { index = i; }

    EventOp get_last_op() const { return last_op; }
    void set_last_op(EventOp op){ last_op = op; }

    void *get_op_obj() const   { return op_obj; }
    void set_op_obj(void *obj) { op_obj = obj; }

    void get_goal(Position &center, int &radius) const {
        center = goal;
        radius = goal_radius;
    }
    void set_goal(Position center, int radius) {
        goal = center;
        goal_radius = radius;
    }

private:
    int id;
    bool dead;
    bool absorbed;

    Position pos;
    Direction dir;
    float hp;

    EventOp last_op;
    void *op_obj;

    Action last_action;
    Reward next_reward, last_reward;
    AgentType &type;
    GroupHandle group;
    int index;

    bool be_involved;

    std::vector<float> embedding;
    Position goal;
    int goal_radius;
};


class Group {
public:
    Group(AgentType &type) : type(type), dead_ct(0), next_reward(0),
                             center_x(0), center_y(0), recursive_base(0) {
    }

    void add_agent(Agent *agent) {
        agents.push_back(agent);
    }

    int get_num()       { return (int)agents.size(); }
    int get_alive_num() { return get_num() - dead_ct; }
    size_t get_size()   { return agents.size(); }

    std::vector<Agent*> &get_agents() { return agents; }
    AgentType &get_type()             { return type; }

    void set_dead_ct(int ct) { dead_ct = ct; }
    int  get_dead_ct() const { return dead_ct; }
    void inc_dead_ct()       { dead_ct++; }

    void clear() {
        agents.clear();
        dead_ct = 0;
    }

    void init_reward() { next_reward = 0; }
    Reward get_reward()         { return next_reward; }
    void add_reward(Reward add) { next_reward += add; }

    // use a base to eliminate duplicates in Gridworld::calc_reward
    int get_recursive_base() {
        return recursive_base;
    }
    void set_recursive_base(int base) { recursive_base = base; }

    void set_center(float cx, float cy) { center_x = cx; center_y = cy; }
    void get_center(float &cx,float &cy) { cx = center_x; cy = center_y; }
    void refresh_center() {
        float sum_x = 0, sum_y = 0;
        for (int i = 0; i < agents.size(); i++) {
            sum_x += agents[i]->get_pos().x;
            sum_y += agents[i]->get_pos().y;
        }
        center_x = sum_x / agents.size();
        center_y = sum_y / agents.size();
    }

private:
    AgentType &type;
    std::vector<Agent*> agents;
    int dead_ct;

    Reward next_reward; // group reward
    float center_x, center_y;

    int recursive_base;
};

struct MoveAction {
    Agent *agent;
    int   action;
};

struct TurnAction {
    Agent *agent;
    int   action;
};

struct AttackAction {
    Agent *agent;
    int   action;
};

} // namespace magent
} // namespace gridworld

#endif //MAGNET_GRIDWORLD_GRIDWORLD_H