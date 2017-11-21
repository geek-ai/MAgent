/**
 * \file runtime_api.cc
 * \brief Runtime library interface
 */

#include "Environment.h"
#include "gridworld/GridWorld.h"
#include "discrete_snake/DiscreteSnake.h"
#include "utility/utility.h"
#include "runtime_api.h"

/**
 *  General Environment
 */
int env_new_game(EnvHandle *game, const char *name) {
    using ::magent::utility::strequ;

    if (strequ(name, "GridWorld")) {
        *game = new ::magent::gridworld::GridWorld();
    } else if (strequ(name, "DiscreteSnake")) {
        *game = new ::magent::discrete_snake::DiscreteSnake();
    } else {
        throw std::invalid_argument("invalid name of game");
    }
    return 0;
}

int env_delete_game(EnvHandle game) {
    LOG(TRACE) << "env delete game.  ";
    delete game;
    return 0;
}

int env_config_game(EnvHandle game, const char *name, void *p_value) {
    LOG(TRACE) << "env config game.  ";
    game->set_config(name, p_value);
    return 0;
}

// run step
int env_reset(EnvHandle game) {
    LOG(TRACE) << "env reset.  ";
    game->reset();
    return 0;
}

int env_get_observation(EnvHandle game, GroupHandle group, float **buffer) {
    LOG(TRACE) << "env get observation.  ";
    game->get_observation(group, buffer);
    return 0;
}

int env_set_action(EnvHandle game, GroupHandle group, const int *actions) {
    LOG(TRACE) << "env set action.  ";
    game->set_action(group, actions);
    return 0;
}

int env_step(EnvHandle game, int *done) {
    LOG(TRACE) << "env step.  ";
    game->step(done);
    return 0;
}

int env_get_reward(EnvHandle game, GroupHandle group, float *buffer) {
    LOG(TRACE) << "env get reward.  ";
    game->get_reward(group, buffer);
    return 0;
}

// info getter
int env_get_info(EnvHandle game, GroupHandle group, const char *name, void *buffer) {
    LOG(TRACE) << "env get info " << name << ".  ";
    game->get_info(group, name, buffer);
    return 0;
}

// render
int env_render(EnvHandle game) {
    LOG(TRACE) << "env render.  ";
    game->render();
    return 0;
}

int env_render_next_file(EnvHandle game) {
    LOG(TRACE) << "env render next file.  ";
    // temporally only needed in DiscreteSnake
    ((::magent::discrete_snake::DiscreteSnake *)game)->render_next_file();
    return 0;
}

/**
 *  GridWorld special
 */
// agent
int gridworld_register_agent_type(EnvHandle game, const char *name, int n,
                                  const char **keys, float *values) {
    LOG(TRACE) << "gridworld register agent type.  ";
    ((::magent::gridworld::GridWorld *)game)->register_agent_type(name, n, keys, values);
    return 0;
}

int gridworld_new_group(EnvHandle game, const char *agent_type_Name, GroupHandle *group) {
    LOG(TRACE) << "gridworld new group.  ";
    ((::magent::gridworld::GridWorld *)game)->new_group(agent_type_Name, group);
    return 0;
}

int gridworld_add_agents(EnvHandle game, GroupHandle group, int n, const char *method,
                         const int *pos_x, const int *pos_y, const int *dir) {
    LOG(TRACE) << "gridworld add agents.  ";
    ((::magent::gridworld::GridWorld *)game)->add_agents(group, n, method, pos_x, pos_y, dir);
    return 0;
}

// run step
int gridworld_clear_dead(EnvHandle game) {
    LOG(TRACE) << "gridworld clear dead.  ";
    ((::magent::gridworld::GridWorld *)game)->clear_dead();
    return 0;
}

int gridworld_set_goal(EnvHandle game, GroupHandle group, const char *method, const int *linear_buffer) {
    LOG(TRACE) << "gridworld clear dead.  ";
    ((::magent::gridworld::GridWorld *)game)->set_goal(group, method, linear_buffer);
    return 0;
}

// reward description
int gridworld_define_agent_symbol(EnvHandle game, int no, int group, int index) {
    LOG(TRACE) << "gridworld define agent symbol";
    ((::magent::gridworld::GridWorld *)game)->define_agent_symbol(no, group, index);
    return 0;
}

int gridworld_define_event_node(EnvHandle game, int no, int op, int *inputs, int n_inputs) {
    LOG(TRACE) << "gridworld define event node";
    ((::magent::gridworld::GridWorld *)game)->define_event_node(no, op, inputs, n_inputs);
    return 0;
}

int gridworld_add_reward_rule(EnvHandle game, int on, int *receiver, float *value, int n_receiver,
                              bool is_terminal, bool auto_value) {
    LOG(TRACE) << "gridworld add reward rule";
    ((::magent::gridworld::GridWorld *)game)->add_reward_rule(on, receiver, value, n_receiver, is_terminal, auto_value);
    return 0;
}

/**
 * DiscreteSnake special
 */
int discrete_snake_add_object(EnvHandle game, int obj_id, int n, const char *method, const int *linear_buffer) {
    LOG(TRACE) << "discrete snake add object.  ";
    ((::magent::discrete_snake::DiscreteSnake *)game)->add_object(obj_id, n, method, linear_buffer);
    return 0;
}

// run step
int discrete_snake_clear_dead(EnvHandle game) {
    LOG(TRACE) << "gridworld clear dead.  ";
    ((::magent::discrete_snake::DiscreteSnake *)game)->clear_dead();
    return 0;
}
