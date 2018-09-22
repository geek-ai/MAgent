/**
 * \file GridWorld.cc
 * \brief core game engine of the gridworld
 */

#include <iostream>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <cassert>

#include "GridWorld.h"

namespace magent {
namespace gridworld {

GridWorld::GridWorld() {
    first_render = true;

    food_mode = false;
    turn_mode = false;
    minimap_mode = false;
    goal_mode = false;
    large_map_mode = false;
    mean_mode = false;

    reward_des_initialized = false;
    embedding_size = 0;
    random_engine.seed(0);

    counter_x = counter_y = nullptr;
}

GridWorld::~GridWorld() {
    for (int i = 0; i < groups.size(); i++) {
        std::vector<Agent*> &agents = groups[i].get_agents();

        // free agents
        size_t agent_size = agents.size();
        #pragma omp parallel for
        for (int j = 0; j < agent_size; j++) {
            delete agents[j];
        }

        // free ranges
        AgentType &type = groups[i].get_type();
        if (type.view_range != nullptr) {
            delete type.view_range;
            type.view_range = nullptr;
        }
        if (type.attack_range != nullptr) {
            delete type.attack_range;
            type.attack_range = nullptr;
        }
        if (type.move_range != nullptr) {
            delete type.move_range;
            type.move_range = nullptr;
        }
    }

    if (counter_x != nullptr)
        delete [] counter_x;
    if (counter_y != nullptr)
        delete [] counter_y;

    if (large_map_mode) {
        delete [] move_buffers;
        delete [] turn_buffers;
    }
}

void GridWorld::reset() {
    id_counter = 0;

    if (width * height > 99 * 99) {
        large_map_mode = true;
        if (width * height > 1000 * 1000) {
            NUM_SEP_BUFFER = 16;
        } else {
            NUM_SEP_BUFFER = 8;
        }
        move_buffers = new std::vector<MoveAction>[NUM_SEP_BUFFER];
        turn_buffers = new std::vector<TurnAction>[NUM_SEP_BUFFER];
    } else
        NUM_SEP_BUFFER = 1;

    // reset map
    map.reset(width, height, food_mode);

    if (counter_x != nullptr)
        delete [] counter_x;
    if (counter_y != nullptr)
        delete [] counter_y;
    counter_x = new int [width];
    counter_y = new int [height];

    render_generator.next_file();
    stat_recorder.reset();

    for (int i = 0;i < groups.size(); i++) {
        std::vector<Agent*> &agents = groups[i].get_agents();

        // free agents
        size_t agent_size = agents.size();
        #pragma omp parallel for
        for (int j = 0; j < agent_size; j++) {
            delete agents[j];
        }

        groups[i].clear();
        groups[i].get_type().n_channel = group2channel((GroupHandle)groups.size());
    }

    if (!reward_des_initialized) {
        init_reward_description();
        reward_des_initialized = true;
    }
}

void GridWorld::set_config(const char *key, void *p_value) {
    float fvalue = *(float *)p_value;
    int ivalue   = *(int *)p_value;
    bool bvalue   = *(bool *)p_value;
    const char *strvalue = (const char *)p_value;

    if (strequ(key, "map_width"))
        width = ivalue;
    else if (strequ(key, "map_height"))
        height = ivalue;

    else if (strequ(key, "food_mode"))      // dead agent will leave food in the map
        food_mode = bvalue;
    else if (strequ(key, "turn_mode"))      // has two more actions -- turn left and turn right
        turn_mode = bvalue;
    else if (strequ(key, "minimap_mode"))   // add minimap into observation
        minimap_mode = bvalue;
    else if (strequ(key, "goal_mode"))      // deprecated every agents has a specific goal
        goal_mode = bvalue;
    else if (strequ(key, "embedding_size")) // embedding size in the observation.feature
        embedding_size = ivalue;

    else if (strequ(key, "render_dir"))     // the directory of saved videos
        render_generator.set_render("save_dir", strvalue);
    else if (strequ(key, "seed"))           // random seed
        random_engine.seed((unsigned long)ivalue);

    else
        LOG(FATAL) << "invalid argument in GridWorld::set_config : " << key;
}

void GridWorld::register_agent_type(const char *name, int n, const char **keys, float *values) {
    std::string str(name);

    if (agent_types.find(str) != agent_types.end())
        LOG(FATAL) << "duplicated name of agent type in GridWorld::register_agent_type : " << str;

    agent_types.insert(std::make_pair(str, AgentType(n, str, keys, values, turn_mode)));
}

void GridWorld::new_group(const char* agent_name, GroupHandle *group) {
    *group = (GroupHandle)groups.size();

    auto iter = agent_types.find(std::string(agent_name));
    if (iter == agent_types.end()) {
        LOG(FATAL) << "invalid name of agent type in new_group : " << agent_name;
    }

    groups.push_back(Group(iter->second));
}

void add_or_error(int ret, int x, int y, int &id_counter, Group &g, Agent *agent) {
    if (ret != 0) {
        LOG(WARNING) << "invalid position in add_agents (" << x << ", " << y << "), already occupied, ignored.\n";
    } else {
        id_counter++;
        g.add_agent(agent);
    }
};

void GridWorld::add_agents(GroupHandle group, int n, const char *method,
                           const int *pos_x, const int *pos_y, const int *pos_dir) {
    int ret;

    if (group == -1) {  // group == -1 for wall
        if (strequ(method, "random")) {
            for (int i = 0; i < n; i++) {
                Position pos = map.get_random_blank(random_engine);
                ret = map.add_wall(pos);
                if  (ret != 0) {
                    LOG(WARNING) << "invalid position in add_wall (" << pos_x[i] << ", " << pos_y[i] << "), "
                                 << "already occupied, ignored.";
                }
            }
        } else if (strequ(method, "custom")) {
            for (int i = 0; i < n; i++) {
                Position pos = Position{pos_x[i], pos_y[i]};
                ret = map.add_wall(pos);
                if (ret != 0) {
                    LOG(WARNING) << "invalid position in add_wall (" << pos_x[i] << ", " << pos_y[i] << "), "
                                 << "already occupied, ignored.";
                }
            }
        } else if (strequ(method, "fill")) {
            // parameter int xs[4] = {x, y, width, height}
            int x_start = pos_x[0],         y_start = pos_x[1];
            int x_end = x_start + pos_x[2], y_end = y_start + pos_x[3];
            for (int x = x_start; x < x_end; x++)
                for (int y = y_start; y < y_end; y++) {
                    ret = map.add_wall(Position{x, y});
                    if (ret != 0) {
                        LOG(WARNING) << "invalid position in add_wall (" << x << ", " << y << "), "
                                     << "already occupied, ignored.";
                    }
                }
        } else {
            LOG(FATAL) << "unsupported method in GridWorld::add_agents : " << method;
        }
    } else {  // group >= 0 for agents
        if (group > groups.size()) {
            LOG(FATAL) << "invalid group handle in GridWorld::add_agents : " << group;
        }
        Group &g = groups[group];
        int base_channel_id = group2channel(group);
        int width = g.get_type().width, length = g.get_type().length;
        AgentType &agent_type = g.get_type();

        if (strequ(method, "random")) {
            for (int i = 0; i < n; i++) {
                Agent *agent = new Agent(agent_type, id_counter, group);
                Direction dir = turn_mode ? (Direction)(random_engine() % DIR_NUM) : NORTH;
                Position pos;

                if (dir == NORTH || dir == SOUTH) {
                    pos = map.get_random_blank(random_engine, width, length);
                } else {
                    pos = map.get_random_blank(random_engine, length, width);
                }

                agent->set_dir(dir);
                agent->set_pos(pos);

                ret = map.add_agent(agent, base_channel_id);
                add_or_error(ret, pos.x, pos.y, id_counter, g, agent);
            }
        } else if (strequ(method, "custom")) {
            for (int i = 0; i < n; i++) {
                Agent *agent = new Agent(agent_type, id_counter, group);

                if (pos_dir[i] >= DIR_NUM) {
                    LOG(FATAL) << "invalid direction in GridWorld::add_agent";
                }

                agent->set_dir(turn_mode ? (Direction) pos_dir[i] : NORTH);
                agent->set_pos((Position) {pos_x[i], pos_y[i]});

                ret = map.add_agent(agent, base_channel_id);
                add_or_error(ret, pos_x[i], pos_y[i], id_counter, g, agent);
            }
        } else if (strequ(method, "fill")) {
            // parameter int xs[4] = {x, y, width, height}
            int x_start = pos_x[0],         y_start = pos_x[1];
            int x_end = x_start + pos_x[2], y_end = y_start + pos_x[3];
            Direction dir = turn_mode ? (Direction)pos_x[4] : NORTH;
            int m_width, m_height;

            if (dir == NORTH || dir == SOUTH) {
                m_width = width;
                m_height = length;
            } else if (dir == WEST || dir == EAST) {
                m_width = length;
                m_height = width;
            } else {
                LOG(FATAL) << "invalid direction in GridWorld::add_agent";
            }

            for (int x = x_start; x < x_end; x += m_width)
                for (int y = y_start; y < y_end; y += m_height) {
                    Agent *agent = new Agent(agent_type, id_counter, group);

                    agent->set_pos(Position{x, y});
                    agent->set_dir(dir);

                    ret = map.add_agent(agent, base_channel_id);
                    add_or_error(ret, x, y, id_counter, g, agent);
                }
        } else {
            LOG(FATAL) << "unsupported method in GridWorld::add_agents : " << method;
        }
    }
}

void GridWorld::get_observation(GroupHandle group, float **linear_buffers) {
    Group &g = groups[group];
    AgentType &type = g.get_type();

    const int n_channel   = g.get_type().n_channel;
    const int view_width  = g.get_type().view_range->get_width();
    const int view_height = g.get_type().view_range->get_height();
    const int n_group = (int)groups.size();
    const int n_action = (int)type.action_space.size();
    const int feature_size = get_feature_size(group);

    std::vector<Agent*> &agents = g.get_agents();
    size_t agent_size = agents.size();

    // transform buffers
    NDPointer<float, 4> view_buffer(linear_buffers[0], {{-1, view_height, view_width, n_channel}});
    NDPointer<float, 2> feature_buffer(linear_buffers[1], {{-1, feature_size}});

    memset(view_buffer.data, 0, sizeof(float) * agent_size * view_height * view_width * n_channel);
    memset(feature_buffer.data, 0, sizeof(float) * agent_size * feature_size);

    // gather view info from AgentType
    const Range *range = type.view_range;
    int view_x_offset = type.view_x_offset, view_y_offset = type.view_y_offset;
    int view_left_top_x, view_left_top_y, view_right_bottom_x, view_right_bottom_y;
    range->get_range_rela_offset(view_left_top_x, view_left_top_y,
                                 view_right_bottom_x, view_right_bottom_y);

    // to make channel layout in observation symmetric to every group
    std::vector<int> channel_trans = make_channel_trans(group,
                                                        group2channel(0),
                                                        type.n_channel,
                                                        n_group);

    // build minimap
    NDPointer<float, 3> minimap(nullptr, {{view_height, view_width, n_group}});
    int scale_h = (height + view_height - 1) / view_height;
    int scale_w = (width + view_width - 1) / view_width;

    if (minimap_mode) {
        minimap.data = new float [view_height * view_width * n_group];
        memset(minimap.data, 0, sizeof(float) * view_height * view_width * n_group);

        std::vector<int> group_sizes;
        for (int i = 0; i < n_group; i++)
            group_sizes.push_back(groups[i].get_size() > 0 ? (int)groups[i].get_size() : 1);

        // by agents
        #pragma omp parallel for
        for (int i = 0; i < n_group; i++) {
            std::vector<Agent*> &agents_ = groups[i].get_agents();
            AgentType type_ = agents[0]->get_type();
            size_t total_ct = 0;
            for (int j = 0; j < agents_.size(); j++) {
                if (type_.can_absorb && agents_[j]->is_absorbed()) // ignore absorbed goal
                    continue;
                Position pos = agents_[j]->get_pos();
                int x = pos.x / scale_w, y = pos.y / scale_h;
                minimap.at(y, x, i)++;
                total_ct++;
            }
            // scale
            for (int j = 0; j < view_height; j++) {
                for (int k = 0; k < view_width; k++) {
                    minimap.at(j, k, i) /= total_ct;
                }
            }
        }
    }

    // fill local view for every agents
    #pragma omp parallel for
    for (int i = 0; i < agent_size; i++) {
        Agent *agent = agents[i];
        // get spatial view
        map.extract_view(agent, view_buffer.data + i*view_height*view_width*n_channel, &channel_trans[0], range,
                         n_channel, view_width, view_height, view_x_offset, view_y_offset,
                         view_left_top_x, view_left_top_y, view_right_bottom_x, view_right_bottom_y);

        if (minimap_mode) {
            int self_x = agent->get_pos().x / scale_w;
            int self_y = agent->get_pos().y / scale_h;
            for (int j = 0; j < n_group; j++) {
                int minimap_channel = channel_trans[group2channel(j)] + 2;
                // copy minimap to channel
                for (int k = 0; k < view_height; k++) {
                    for (int l = 0; l < view_width; l++) {
                        view_buffer.at(i, k, l, minimap_channel)= minimap.at(k, l, j);
                    }
                }
                view_buffer.at(i, self_y, self_x, minimap_channel) += 1;
            }
        }

        // get non-spatial feature
        agent->get_embedding(feature_buffer.data + i*feature_size, embedding_size);
        Position pos = agent->get_pos();
        // last action
        feature_buffer.at(i, embedding_size + agent->get_action()) = 1;
        // last reward
        feature_buffer.at(i, embedding_size + n_action) = agent->get_last_reward();
        if (minimap_mode) { // absolute coordination
            feature_buffer.at(i, embedding_size + n_action + 1) = (float) pos.x / width;
            feature_buffer.at(i, embedding_size + n_action + 2) = (float) pos.y / height;
        }
    }

    if (minimap_mode)
        delete [] minimap.data;
}

void GridWorld::set_action(GroupHandle group, const int *actions) {
    std::vector<Agent*> &agents = groups[group].get_agents();
    const AgentType &type = groups[group].get_type();
    // action space layout : move turn attack ...
    const int bandwidth = (width + NUM_SEP_BUFFER - 1) / NUM_SEP_BUFFER;

    size_t agent_size = agents.size();

    if (large_map_mode) { // divide actions in group, in order to compute them in parallel
        for (int i = 0; i < agent_size; i++) {
            Agent *agent = agents[i];
            Action act = (Action) actions[i];
            agent->set_action(act);

            if (act < type.turn_base) {          // move
                int x = agent->get_pos().x;
                int x_ = x % bandwidth;
                if (x_ < 4 || x_ > bandwidth - 4) {
                    move_buffer_bound.push_back(MoveAction{agent, act - type.move_base});
                } else {
                    int to = agent->get_pos().x / bandwidth;
                    move_buffers[to].push_back(MoveAction{agent, act - type.move_base});
                }
            } else if (act < type.attack_base) { // turn
                int x = agent->get_pos().x;
                int x_ = x % bandwidth;
                if (x_ < 4 || x_ > bandwidth - 4) {
                    turn_buffer_bound.push_back(TurnAction{agent, act - type.move_base});
                } else {
                    int to = agent->get_pos().x / bandwidth;
                    turn_buffers[to].push_back(TurnAction{agent, act - type.move_base});
                }
            } else {                             // attack
                attack_buffer.push_back(AttackAction{agent, act - type.attack_base});
            }
        }
    } else {
        for (int i = 0; i < agent_size; i++) {
            Agent *agent = agents[i];
            Action act = (Action) actions[i];
            agent->set_action(act);

            if (act < type.turn_base) {          // move
                move_buffer_bound.push_back(MoveAction{agent, act - type.move_base});
            } else if (act < type.attack_base) { // turn
                turn_buffer_bound.push_back(TurnAction{agent, act - type.move_base});
            } else {                             // attack
                attack_buffer.push_back(AttackAction{agent, act - type.attack_base});
            }
        }
    }
}

void GridWorld::step(int *done) {
    #pragma omp declare reduction (merge : std::vector<RenderAttackEvent> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
    const bool stat = false;

    LOG(TRACE) << "gridworld step begin.  ";
    size_t attack_size = attack_buffer.size();
    size_t group_size  = groups.size();

    // shuffle attacks
    for (int i = 0; i < attack_size; i++) {
        int j = (int)random_engine() % (i+1);
        std::swap(attack_buffer[i], attack_buffer[j]);
    }

    LOG(TRACE) << "attack.  ";
    std::vector<RenderAttackEvent> render_attack_buffer;
    std::map<PositionInteger, int> attack_obj_counter;    // for statistic info

    // attack
    #pragma omp parallel for reduction(merge: render_attack_buffer)
    for (int i = 0; i < attack_size; i++) {
        Agent *agent = attack_buffer[i].agent;

        if (agent->is_dead())
            continue;

        int obj_x, obj_y;
        PositionInteger obj_pos = map.get_attack_obj(attack_buffer[i], obj_x, obj_y);
        if (!first_render)
            render_attack_buffer.emplace_back(RenderAttackEvent{agent->get_id(), obj_x, obj_y});

        if (obj_pos == -1) {  // attack blank block
            agent->add_reward(agent->get_type().attack_penalty);
            continue;
        }

        if (stat) {
            attack_obj_counter[obj_pos]++;
        }

        float reward = 0.0;
        GroupHandle dead_group = -1;
        #pragma omp critical
        {
            reward = map.do_attack(agent, obj_pos, dead_group);
            if (dead_group != -1) {
                groups[dead_group].inc_dead_ct();
            }
        }
        agent->add_reward(reward + agent->get_type().attack_penalty);
    }
    attack_buffer.clear();
    if (!first_render)
        render_generator.set_attack_event(render_attack_buffer);

    if (stat) {
        for (auto iter : attack_obj_counter) {
            if (iter.second > 1) {
                stat_recorder.both_attack++;
            }
        }
    }

    // starve
    LOG(TRACE) << "starve.  ";
    for (int i = 0; i < group_size; i++) {
        Group &group = groups[i];
        std::vector<Agent*> &agents = group.get_agents();
        int starve_ct = 0;
        size_t agent_size = agents.size();

        #pragma omp parallel for reduction(+: starve_ct)
        for (int j = 0; j < agent_size; j++) {
            Agent *agent = agents[j];

            if (agent->is_dead())
                continue;

            // alive agents
            bool starve = agent->starve();
            if (starve) {
                map.remove_agent(agent);
                starve_ct++;
            }
        }
        group.set_dead_ct(group.get_dead_ct() + starve_ct);
    }

    if (turn_mode) {
        // do turn
        auto do_turn_for_a_buffer = [] (std::vector<TurnAction> &turn_buf, Map &map) {
            //std::random_shuffle(turn_buf.begin(), turn_buf.end());
            size_t turn_size = turn_buf.size();
            for (int i = 0; i < turn_size; i++) {
                Action act = turn_buf[i].action;
                Agent *agent = turn_buf[i].agent;

                if (agent->is_dead())
                    continue;

                int dir = act * 2 - 1;
                map.do_turn(agent, dir);
            }
            turn_buf.clear();
        };

        if (large_map_mode) {
            LOG(TRACE) << "turn parallel.  ";
            #pragma omp parallel for
            for (int i = 0; i < NUM_SEP_BUFFER; i++) {        // turn in separate areas, do them in parallel
                do_turn_for_a_buffer(turn_buffers[i], map);
            }
        }
        LOG(TRACE) << "turn boundary.   ";
        do_turn_for_a_buffer(turn_buffer_bound, map);
    }

    // do move
    auto do_move_for_a_buffer = [] (std::vector<MoveAction> &move_buf, Map &map) {
        //std::random_shuffle(move_buf.begin(), move_buf.end());
        size_t move_size = move_buf.size();
        for (int j = 0; j < move_size; j++) {
            Action act = move_buf[j].action;
            Agent *agent = move_buf[j].agent;

            if (agent->is_dead() || agent->is_absorbed())
                continue;

            int dx, dy;
            int delta[2];
            agent->get_type().move_range->num2delta(act, dx, dy);
            switch(agent->get_dir()) {
                case NORTH:
                    delta[0] = dx;  delta[1] = dy;  break;
                case SOUTH:
                    delta[0] = -dx; delta[1] = -dy; break;
                case WEST:
                    delta[0] = dy;  delta[1] = -dx; break;
                case EAST:
                    delta[0] = -dy; delta[1] = dx;  break;
                default:
                    LOG(FATAL) << "invalid direction in GridWorld::step when do move";
            }

            map.do_move(agent, delta);
        }
        move_buf.clear();
    };

    if (large_map_mode) {
        LOG(TRACE) << "move parallel.  ";
        #pragma omp parallel for
        for (int i = 0; i < NUM_SEP_BUFFER; i++) {    // move in separate areas, do them in parallel
            do_move_for_a_buffer(move_buffers[i], map);
        }
    }
    LOG(TRACE) << "move boundary.  ";
    do_move_for_a_buffer(move_buffer_bound, map);

    LOG(TRACE) << "calc_reward.  ";
    calc_reward();

    LOG(TRACE) << "game over check.  ";
    int live_ct = 0;  // default game over condition: all the agents in an arbitrary group die
    for (int i = 0; i < groups.size(); i++) {
        if (groups[i].get_alive_num() > 0)
            live_ct++;
    }
    *done = (int)(live_ct < groups.size());

    size_t rule_size = reward_rules.size();
    for (int i = 0; i < rule_size; i++) {
        if (reward_rules[i].trigger && reward_rules[i].is_terminal)
            *done = (int)true;
    }
}

void GridWorld::clear_dead() {
    size_t group_size = groups.size();

    #pragma omp parallel for
    for (int i = 0; i < group_size; i++) {
        Group &group = groups[i];
        group.init_reward();
        std::vector<Agent*> &agents = group.get_agents();

        // clear dead agents
        size_t agent_size = agents.size();
        int dead_ct = 0;
        unsigned int pt = 0;
        float sum_x = 0, sum_y = 0;

        for (int j = 0; j < agent_size; j++) {
            Agent *agent = agents[j];
            if (agent->is_dead()) {
                delete agent;
                dead_ct++;
            } else {
                agent->init_reward();
                agent->set_index(pt);
                agents[pt++] = agent;

                //Position pos = agent->get_pos();
                //sum_x += pos.x; sum_y += pos.y;
            }
        }
        agents.resize(pt);
        group.set_dead_ct(0);
    }
}

void GridWorld::set_goal(GroupHandle group, const char *method, const int *linear_buffer) {
    // deprecated
    if (strequ(method, "random")) {
        std::vector<Agent*> &agents = groups[group].get_agents();
        for (int i = 0; i < agents.size(); i++) {
            int x = (int)random_engine() % width;
            int y = (int)random_engine() % height;
            agents[i]->set_goal(Position{x, y}, 0);
        }
    } else {
        LOG(FATAL) << "invalid goal type in GridWorld::set_goal";
    }
}

void GridWorld::calc_reward() {
    size_t rule_size = reward_rules.size();
    for (int i = 0; i < groups.size(); i++)
        groups[i].set_recursive_base(0);

    for (int i = 0; i < rule_size; i++) {
        reward_rules[i].trigger = false;
        std::vector<AgentSymbol*> &input_symbols = reward_rules[i].input_symbols;
        std::vector<AgentSymbol*> &infer_obj     = reward_rules[i].infer_obj;
        calc_rule(input_symbols, infer_obj, reward_rules[i], 0);
    }
}

void GridWorld::get_reward(GroupHandle group, float *buffer) {
    std::vector<Agent*> &agents = groups[group].get_agents();

    size_t  agent_size = agents.size();
    Reward  group_reward = groups[group].get_reward();

    #pragma omp parallel for
    for (int i = 0; i < agent_size; i++) {
        buffer[i] = agents[i]->get_reward() + group_reward;
    }
}

/**
 * info getter
 */
void GridWorld::get_info(GroupHandle group, const char *name, void *void_buffer) {
    // for more information from the engine, add items here

    std::vector<Agent*> &agents = groups[group].get_agents();
    int   *int_buffer   = (int *)void_buffer;
    float *float_buffer = (float *)void_buffer;
    bool  *bool_buffer  = (bool *)void_buffer;

    if (strequ(name, "num")) {         // int
        int_buffer[0] = groups[group].get_num();
    } else if (strequ(name, "id")) {   // int
        size_t agent_size = agents.size();
        #pragma omp parallel for
        for (int i = 0; i < agent_size; i++) {
            int_buffer[i] = agents[i]->get_id();
        }
    } else if (strequ(name, "pos")) {   // int
        size_t agent_size = agents.size();
        #pragma omp parallel for
        for (int i = 0; i < agent_size; i++) {
            int_buffer[2 * i] = agents[i]->get_pos().x;
            int_buffer[2 * i + 1] = agents[i]->get_pos().y;
        }
    } else if (strequ(name, "alive")) {  // bool
        size_t agent_size = agents.size();
        #pragma omp parallel for
        for (int i = 0; i < agent_size; i++) {
            bool_buffer[i] = !agents[i]->is_dead();
        }
    } else if (strequ(name, "global_minimap")) {
        size_t n_group = groups.size();

        int view_height = (int)lround(float_buffer[0]);
        int view_width = (int)lround(float_buffer[1]);
        memset(float_buffer, 0, sizeof(float) * view_height * view_width * n_group);

        NDPointer<float, 3> minimap(float_buffer, {view_height, view_width, (int)n_group});

        int scale_h = (height + view_height - 1) / view_height;
        int scale_w = (width + view_width - 1) / view_width;

        for (size_t i = 0; i < n_group; i++) {
            size_t channel = (i - group + n_group) % n_group;
            std::vector<Agent*> &agents_ = groups[i].get_agents();
            for (size_t j = 0; j < agents_.size(); j++) {
                Position pos = agents_[j]->get_pos();
                int x = pos.x / scale_w, y = pos.y / scale_h;
                minimap.at(y, x, channel)++;
            }
            // scale
            for (size_t j = 0; j < view_height; j++) {
                for (size_t k = 0; k < view_width; k++) {
                    minimap.at(j, k, channel) /= agents_.size();
                }
            }
        }
    } else if (strequ(name, "mean_info")) {
        size_t agent_size = agents.size();
        int n_action = (int)groups[group].get_type().action_space.size();
        int *action_counter = new int[n_action];
        float sum_x, sum_y;
        sum_x = sum_y = 0;
        memset(action_counter, 0, sizeof(int) * n_action);
        #pragma omp parallel for reduction(+: sum_x) reduction(+: sum_y)
        for (int i = 0; i < agent_size; i++) {
            Position pos = agents[i]->get_pos();
            sum_x += pos.x;
            sum_y += pos.y;
            int x = agents[i]->get_action();
            #pragma omp atomic
            action_counter[x]++;
        }

        assert (agent_size != 0);
        float_buffer[0] = sum_x / agent_size;
        float_buffer[1] = sum_y / agent_size;
        for (int i = 0; i < n_action; i++)
            float_buffer[2 + i] = (float)(1.0 * action_counter[i] / agent_size);
    } else if (strequ(name, "walls_info")) {
        std::vector<Position> walls;
        map.get_wall(walls);

        NDPointer<int, 2> coor(int_buffer, {-1, 2});
        for (int i = 0; i < walls.size(); i++) {
            coor.at(i+1, 0) = walls[i].x;
            coor.at(i+1, 1) = walls[i].y;
        }
        coor.at(0, 0) = (int)walls.size();
    } else if (strequ(name, "render_window_info")) {
        first_render = false;

        int ct = 1;
        int range_x1, range_y1, range_x2, range_y2;

        // read parameter
        range_x1 = int_buffer[0];
        range_y1 = int_buffer[1];
        range_x2 = int_buffer[2];
        range_y2 = int_buffer[3];

        // fill valid agents
        NDPointer<int, 2> ret(int_buffer, {-1, 4});
        for (int i = 0; i < groups.size(); i++) {
            std::vector<Agent *> &agents = groups[i].get_agents();
            for (int j = 0; j < agents.size(); j++) {
                int id = agents[j]->get_id();
                Position pos = agents[j]->get_pos();

                if (pos.x < range_x1 || pos.x > range_x2 || pos.y < range_y1 ||
                    pos.y > range_y2)
                    continue;

                if (agents[j]->get_type().can_absorb && !agents[j]->is_absorbed())
                    continue;

                ret.at(ct, 0) = id;
                ret.at(ct, 1) = pos.x;
                ret.at(ct, 2) = pos.y;
                ret.at(ct, 3) = i;
                ct++;
            }
        }

        // the first line of ret returns counter info
        ret.at(0, 0) = ct - 1;
        ret.at(0, 1) = (int)render_generator.get_attack_event().size();
    } else if (strequ(name, "attack_event")) {
        std::vector<RenderAttackEvent> &attack_events = render_generator.get_attack_event();
        NDPointer<int, 2> ret(int_buffer, {-1, 3});
        for (int i = 0; i < attack_events.size(); i++) {
            ret.at(i, 0) = attack_events[i].id;
            ret.at(i, 1) = attack_events[i].x;
            ret.at(i, 2) = attack_events[i].y;
        }
    } else if (strequ(name, "action_space")) {  // int
        int_buffer[0] = (int)groups[group].get_type().action_space.size();
    } else if (strequ(name, "view_space")) {    // int
        // the following is necessary! user can call get_view_space before reset
        groups[group].get_type().n_channel = group2channel((GroupHandle)groups.size());
        int_buffer[0] = groups[group].get_type().view_range->get_height();
        int_buffer[1] = groups[group].get_type().view_range->get_width();
        int_buffer[2] = groups[group].get_type().n_channel;
    } else if (strequ(name, "feature_space")) {
        int_buffer[0] = get_feature_size(group);
    } else if (strequ(name, "view2attack")) {
        const AgentType &type = groups[group].get_type();
        const Range *range = type.attack_range;
        const Range *view_range = type.view_range;
        const int view_width = view_range->get_width(), view_height = view_range->get_height();

        NDPointer<int, 2> ret(int_buffer, {view_height, view_width});
        memset(ret.data, -1, sizeof(int) * view_height * view_width);
        int x1, y1, x2, y2;

        view_range->get_range_rela_offset(x1, y1, x2, y2);
        for (int i = 0; i < range->get_count(); i++) {
            int dx, dy;
            range->num2delta(i, dx, dy);
            //dx -= type.att_x_offset; dy -= type.att_y_offset;

            ret.at(dy - y1, dx - x1) = i;
        }
    } else if (strequ(name, "attack_base")) {
        int_buffer[0] = groups[group].get_type().attack_base;
    }  else if (strequ(name, "groups_info")) {
        const int colors[][3] = {
                {192, 64, 64},
                {64, 64, 192},
                {64, 192, 64},
                {64, 64, 64},
        };
        NDPointer<int, 2> info(int_buffer, {-1, 5});

        for (int i = 0; i < groups.size(); i++) {
            info.at(i, 0) = groups[i].get_type().width;
            info.at(i, 1) = groups[i].get_type().length;
            info.at(i, 2) = colors[i][0];
            info.at(i, 3) = colors[i][1];
            info.at(i, 4) = colors[i][2];
        }
    } else if (strequ(name, "both_attack")) {
        int_buffer[0] = stat_recorder.both_attack;
    } else {
        LOG(FATAL) << "unsupported info name in GridWorld::get_info : " << name;
    }
}

// private utility
std::vector<int> GridWorld::make_channel_trans(
        GroupHandle group,
        int base, int n_channel, int n_group) {
    std::vector<int> trans((unsigned int)n_channel);
    for (int i = 0; i < base; i++)
        trans[i] = i;
    for (int i = 0; i < groups.size(); i++) {
        int cycle_group = (group + i) % n_group;
        trans[group2channel(cycle_group)] = base;
        if (minimap_mode) {
            base += 3;
        } else {
            base += 2;
        }
    }
    return trans;
}

int GridWorld::group2channel(GroupHandle group) {
    int base = 1;
    int scale = 2;
    if (food_mode)
        base++;
    if (minimap_mode)
        scale++;

    return base + group * scale; // wall + additional + (has, hp) + (has, hp) + ...
}

int GridWorld::get_feature_size(GroupHandle group) {
    // feature space layout : [embedding, last_action (one hot), last_reward]
    int feature_space = embedding_size + (int)groups[group].get_type().action_space.size() + 1;
    if (goal_mode)
        feature_space += 2;
    if (minimap_mode)  // x, y coordinate
        feature_space += 2;
    return feature_space;
}

/**
 * render
 */
void GridWorld::render() {
    if (render_generator.get_save_dir() == "___debug___")
        map.render();
    else {
        if (first_render) {
            first_render = false;
            render_generator.gen_config(groups, width, height);
        }
        render_generator.render_a_frame(groups, map);
    }
}

} // namespace magent
} // namespace gridworld

// reward counter for align
//    memset(counter_x, 0, width * sizeof(int));
//    memset(counter_y, 0, height * sizeof(int));
//    for (int i = 0; i < group_size; i++) {
//        std::vector<Agent*> &agents = groups[i].get_agents();
//        size_t agent_size = agents.size();
//        #pragma omp parallel for
//        for (int j = 0; j < agent_size; j++) {
//            if (agents[j]->is_dead())
//                continue;
//            Position pos = agents[j]->get_pos();
//            #pragma omp atomic
//            counter_x[pos.x]++;
//            #pragma omp atomic
//            counter_y[pos.y]++;
//        }
//    }