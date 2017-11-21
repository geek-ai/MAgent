/**
 * \file DiscreteSnake.cc
 * \brief core game engine of the discrete snake
 */

#include <stdexcept>
#include <cstring>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include "DiscreteSnake.h"


namespace magent {
namespace discrete_snake {

DiscreteSnake::DiscreteSnake() {
    width = height = 100;
    view_width = view_height = 21;
    total_resource = (int)(100 * 100 * 0.1);
    embedding_size = 16;
    max_dead_penalty = -10;
    corpse_value = 1;
    initial_length = 3;
    head_mask = nullptr;

    first_render = true;
}

DiscreteSnake::~DiscreteSnake() {
    if (head_mask != nullptr)
        delete head_mask;

    for (auto agent : agents)
        delete agent;

    for (auto food : foods)
        delete food;
}

void DiscreteSnake::reset() {
    id_counter = 0;
    render_generator.next_file();

    map.reset(width, height);
    head_mask = new int [width * height];

    //free agents
    for (int i = 0; i < agents.size(); i++) {
        delete agents[i];
    }
}

void DiscreteSnake::set_config(const char *key, void *p_value) {
    float fvalue = *(float *)p_value;
    int ivalue   = *(int *)p_value;
    bool bvalue   = *(bool *)p_value;
    const char *strvalue = (const char *)p_value;

    if (strequ(key, "map_width"))
        width = ivalue;
    else if (strequ(key, "map_height"))
        height = ivalue;

    else if (strequ(key, "view_width"))
        view_width = ivalue;
    else if (strequ(key, "view_height"))
        view_height = ivalue;
    else if (strequ(key, "max_dead_penalty"))
        max_dead_penalty = fvalue;
    else if (strequ(key, "corpse_value"))
        corpse_value = fvalue;
    else if (strequ(key, "initial_length"))
        initial_length = ivalue;
    else if (strequ(key, "total_resource"))
        total_resource = ivalue;

    else if (strequ(key, "embedding_size"))
        embedding_size = ivalue;
    else if (strequ(key, "render_dir"))
        render_generator.set_render("save_dir", strvalue);

    else if (strequ(key, "seed"))
        srand(ivalue);

    else
        LOG(FATAL) << "invalid argument in DiscreteSnake::set_config: " << key;
}

void DiscreteSnake::add_object(int obj_id, int n, const char *method, const int *linear_buffer) {
    if (obj_id == -1) {  // wall

    } else if (obj_id == -2) { // food
        std::vector<Position> pos;
        if (strequ(method, "random")) { // snake
            for (int i = 0; i < n; i++) {
                Food *food = new Food(1, 1, corpse_value);

                map.get_random_blank(pos, 1);
                foods.insert(food);
                map.add_food(food, pos[0].x, pos[0].y);
            }
        } else {
            LOG(FATAL) << "unsupported method in DiscreteSnake::add_object : " << method;
        }
    } else if (obj_id == 0) { // snake
        if (strequ(method, "random")) { // snake
            std::vector<Position> pos;
            for (int i = 0; i < n; i++) {
                Agent *agent = new Agent(id_counter);
                Direction dir = (Direction) (random() % (int) DIR_NUM);

                map.get_random_blank(pos, initial_length);

                agent->set_dir(dir);
                agent->init_body(pos);

                map.add_agent(agent);
                agents.push_back(agent);
            }
        } else {
            LOG(FATAL) << "unsupported method in DiscreteSnake::add_object : " << method;
        }
    }
}

void DiscreteSnake::get_observation(GroupHandle group, float **linear_buffer) {
    int n_channel = CHANNEL_NUM; // wall food self other id
    int n_action = (int)ACT_NUM;
    int feature_size = embedding_size + n_action + 1; // embedding + last_action + length

    float (*view_buffer)[view_height][view_width][n_channel];
    float (*feature_buffer)[feature_size];

    view_buffer = (decltype(view_buffer))linear_buffer[0];
    feature_buffer = (decltype(feature_buffer))linear_buffer[1];

    size_t agent_size = agents.size();

    memset(view_buffer, 0, sizeof(float) * agent_size * view_height * view_width * n_channel);
    memset(feature_buffer, 0, sizeof(float) * agent_size * feature_size);

    #pragma omp parallel for
    for (int i = 0; i < agent_size; i++) {
        Agent *agent = agents[i];

        map.extract_view(agent, (float *)view_buffer[i],
                         view_height, view_width, n_channel, id_counter);
        agent->get_embedding(feature_buffer[i], embedding_size);
        feature_buffer[i][embedding_size + agent->get_action()] = 1;
        feature_buffer[i][embedding_size + n_action] = agent->get_length();
    }
}

void DiscreteSnake::set_action(GroupHandle group, const int *actions) {
    #pragma omp parallel for
    for (int i = 0; i < agents.size(); i++) {
        Agent *agent = agents[i];
        Action act = (Action) actions[i];
        agent->set_action(act);
    }
}

void DiscreteSnake::step(int *done) {
    #pragma omp declare reduction (merge : std::vector<Agent*> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
    #pragma omp declare reduction (merge : std::vector<Food*>  : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
    #pragma omp declare reduction (merge : std::set<PositionInteger> : omp_out.insert(omp_in.begin(), omp_in.end()))

    const Action dir2inverse[] = {
        ACT_LEFT, ACT_UP, ACT_RIGHT, ACT_DOWN,
    };

    const int delta[][2] = {
        {1, 0}, {0, 1}, {-1, 0}, {0, -1},
    };

    const double eps = 1e-6;

    // update body
    LOG(TRACE) << "update body.  ";
    size_t agent_size = agents.size();
    #pragma omp parallel for
    for (int i = 0; i < agent_size; i++) {
        Agent *agent = agents[i];

        Action act = agent->get_action();
        Direction dir = agent->get_dir();

        if (act != ACT_NOOP && (int)act != (int)dir && act != dir2inverse[dir]) {
            dir = (Direction)act;
            agent->set_dir(dir);
        }

        // push new head
        Position head = agent->get_head();
        head.x += delta[dir][0];
        head.y += delta[dir][1];
        agent->push_head(head);

        // pop old tail
        if (agent->get_total_reward() + 1 + initial_length - eps < agent->get_length())
            map.move_tail(agent);
    }

    memset(head_mask, 0, sizeof(int) * width * height);
    for (int i = 0; i < agent_size; i++)
        head_mask[map.pos2int(agents[i]->get_head())]++;

    // check head (food or wall or other)
    LOG(TRACE) << "check head.  ";
    std::vector<Food*>  eat_list;
    std::vector<Agent*> dead_list;
    std::set<PositionInteger> double_head_list;

    int added_length = 0;
    #pragma omp parallel for reduction(merge: dead_list) reduction(merge: eat_list) reduction(merge: double_head_list) reduction(+: added_length)
    for (int i = 0; i < agent_size; i++) {
        Agent *agent = agents[i];
        Food *eaten = nullptr;

        float reward = 0;
        bool dead = false;

        PositionInteger head_int = map.pos2int(agent->get_head());
        if (head_mask[head_int] > 1) {
            dead = true;
            double_head_list.insert(head_int);
        } else {
            map.move_head(agent, head_int, reward, dead, eaten);
        }

        if (dead) {
            dead_list.push_back(agent);
            agent->set_dead();
            //agent->add_reward(-std::min(-max_dead_penalty, (Reward)(agent->get_length() - initial_length)));
            agent->add_reward(-max_dead_penalty);
        } else {
            if (eaten != nullptr) {
                eat_list.push_back(eaten);
                agent->add_reward(reward);
            }
            // calc total length for resource balancing
            added_length += agent->get_length() - initial_length;
        }
    }

    // delete eaten foods
    LOG(TRACE) << "delete eaten food.  ";
    #pragma omp parallel for
    for (int i = 0; i < eat_list.size(); i++) {
        map.remove_food(eat_list[i]);
        delete eat_list[i];
    }
    for (int i = 0; i < eat_list.size(); i++) {
        foods.erase(eat_list[i]);
    }

    // make dead agents as food
    LOG(TRACE) << "make food.  ";
    std::vector<Food*> new_foods;
    #pragma omp parallel for reduction(merge: new_foods)
    for (int i = 0; i < dead_list.size(); i++) {
        int add = (int)dead_list[i]->get_length() - initial_length;
        map.make_food(dead_list[i], corpse_value, new_foods, add);
    }
    foods.insert(new_foods.begin(), new_foods.end());

    // double head, balance total resource
    int add = total_resource - added_length - (int)foods.size();
    if (add > 0) {
        for (auto pos_int : double_head_list) {
            Food *food = new Food(1, 1, corpse_value);
            Position pos = map.int2pos(pos_int);

            if (map.add_food(food, pos.x, pos.y)) {
                foods.insert(food);
                if (--add == 0)
                    break;
            }
        }
    }
    add_object(-2, add, "random", nullptr);

    // find new position for dead agents
    /*LOG(TRACE) << "find new position.  ";
    for (int i = 0; i < dead_list.size(); i++) {
        Agent *agent = dead_list[i];
        Direction dir = (Direction)(random() % (int)DIR_NUM);
        std::vector<Position> pos;

        bool found = map.get_random_blank(pos, initial_length);
        if (!found)
            LOG(FATAL) << "filled map";

        agent->set_dir(dir);
        agent->init_body(pos);
        map.add_agent(agent);
    }*/

    /*if (foods.size() != map.get_food_num()) {
        printf("%d %d\n", foods.size(), map.get_food_num());
        exit(0);
    }*/

    *done = 0;
}

void DiscreteSnake::get_reward(GroupHandle group, float *buffer) {
    size_t agent_size = agents.size();
    #pragma omp parallel for
    for (int i = 0; i < agent_size; i++) {
        buffer[i] = agents[i]->get_reward();
    }
}

void DiscreteSnake::clear_dead() {
    size_t pt = 0;
    size_t agent_size = agents.size();
    for (int j = 0; j < agent_size; j++) {
        Agent *agent = agents[j];
        if (agent->is_dead()) {
            delete agent;
        } else {
            agent->init_reward();
            agents[pt++] = agent;

        }
    }
    agents.resize(pt);
}

/**
 * info getter
 */
void DiscreteSnake::get_info(GroupHandle group, const char *name, void *void_buffer) {
    int   *int_buffer   = (int *)void_buffer;
    float *float_buffer = (float *)void_buffer;
    bool  *bool_buffer  = (bool *)void_buffer;

    if (strequ(name, "id")) {
        size_t agent_size = agents.size();
        #pragma omp parallel for
        for (int i = 0; i < agent_size; i++) {
            int_buffer[i] = agents[i]->get_id();
        }
    } else if (strequ(name, "num")) {
        if (group == 0) {
            int_buffer[0] = (int) agents.size();
        } else if (group == -1) { // wall
            int_buffer[0] = 0;
        } else if (group == -2) { // food
            int_buffer[0] = (int) foods.size();
        }
    } else if (strequ(name, "length")) {
        size_t agent_size = agents.size();
        #pragma omp parallel for
        for (int i = 0; i < agent_size; i++) {
            int_buffer[i] = (int) agents[i]->get_length();
        }
    } else if (strequ(name, "alive")) {
        size_t agent_size = agents.size();
        #pragma omp parallel for
        for (int i = 0; i < agent_size; i++) {
            bool_buffer[i] = !agents[i]->is_dead();
        }
    } else if (strequ(name, "head")) {
        size_t agent_size = agents.size();
        #pragma omp parallel for
        for (int i = 0; i < agent_size; i++) {
            int_buffer[2 * i] = agents[i]->get_body().front().x;
            int_buffer[2 * i + 1] = agents[i]->get_body().front().y;
        }
    } else if (strequ(name, "action_space")) {
        int_buffer[0] = (int)ACT_NUM;
    } else if (strequ(name, "view_space")) {
        int_buffer[0] = view_height;
        int_buffer[1] = view_width;
        int_buffer[2] = CHANNEL_NUM;
    } else if (strequ(name, "feature_space")) {
        int n_action = (int)ACT_NUM;
        int_buffer[0] = embedding_size + n_action + 1; // embedding + last_action + length
    } else {
        std::cerr << name << std::endl;
        LOG(FATAL) << "unsupported info name in DiscreteSnake::get_info : " << name;
    }
}

/**
 * render
 */
void DiscreteSnake::render() {
    if (first_render) {
        first_render = false;
        render_generator.gen_config(map, width, height);
    }

    render_generator.render_a_frame(agents, foods);
}

void DiscreteSnake::render_next_file(){
    render_generator.next_file();
}

} // namespace discrete_snake
} // namespace magent