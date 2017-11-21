/**
 * \file runtime_api.h
 * \brief Runtime library interface
 */

#ifndef MAGENT_RUNTIME_API_H
#define MAGENT_RUNTIME_API_H

#include "Environment.h"

extern "C" {

using ::magent::environment::EnvHandle;
using ::magent::environment::GroupHandle;

/**
 *  General Environment
 */
// game
int env_new_game(EnvHandle *game, const char *name);
int env_delete_game(EnvHandle game);
int env_config_game(EnvHandle game, const char *name, void *p_value);

// run step
int env_reset(EnvHandle game);
int env_get_observation(EnvHandle game, GroupHandle group, float **buffer);
int env_set_action(EnvHandle game, GroupHandle group, const int *actions);
int env_step(EnvHandle game, int *done);
int env_get_reward(EnvHandle game, GroupHandle group, float *buffer);

// info getter
int env_get_info(EnvHandle game, GroupHandle group, const char *name, void *buffer);

// render
int env_render(EnvHandle game);
int env_render_next_file(EnvHandle game);

/**
 *  GridWorld special
 */
// agent
int gridworld_register_agent_type(EnvHandle game, const char *name, int n, const char **keys, float *values);
int gridworld_new_group(EnvHandle game, const char *agent_type_name, GroupHandle *group);
int gridworld_add_agents(EnvHandle game, GroupHandle group, int n, const char *method,
                         const int *pos_x, const int *pos_y, const int *dir);

// run step
int gridworld_clear_dead(EnvHandle game);
int gridworld_set_goal(EnvHandle game, GroupHandle group, const char *method, const int *linear_buffer);

// reward description
int gridworld_define_agent_symbol(EnvHandle game, int no, int group, int index);
int gridworld_define_event_node(EnvHandle game, int no, int op, int *inputs, int n_inputs);
int gridworld_add_reward_rule(EnvHandle game, int on, int *receiver, float *value, int n_receiver,
                              bool is_terminal, bool auto_value);

/**
 *  DiscreteSnake special
 */
int discrete_snake_clear_dead(EnvHandle game);
int discrete_snake_add_object(EnvHandle game, int obj_id, int n, const char *method, const int *linear_buffer);


/**
 * Temporary C Booster
 */
void runaway_infer_action(float *obs_buf, float *feature_buf, int n, int height, int width, int n_channel,
                          int attack_base, int *act_buf, int away_channel, int move_back);
void rush_prey_infer_action(float *obs_buf, float *feature_buf, int n, int height,  int width, int n_channel,
                            int *act_buf, int attack_channel, int attack_base,
                            int *view2attack_buf, float threshold);
void gather_infer_action(float *obs_buf, float *hp_buf, int n, int height, int width, int n_channel,
                         int *act_buf, int attack_base, int *view2attack_buf);
}

#endif // MAGENT_RUNTIME_API_H
