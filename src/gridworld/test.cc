/**
 * \file test.cc
 * \brief unit test for some function
 */

#if 0
#include "../Environment.h"
#include "GridWorld.h"
#include "../runtime_api.h"

using ::magent::gridworld::SectorRange;
using ::magent::gridworld::CircleRange;

void test_sector_range() {
    printf(" ==== test sector range ==== \n");

    /*SectorRange s1(120, 3, 0);
    SectorRange s2(120, 5, 0);

    s1.print_self();
    s2.print_self();*/

    CircleRange r1(4, 1, 0);
    CircleRange r2(4, 0, 1);

    r1.print_self();
    r2.print_self();
}

void get_observation(EnvHandle game, GroupHandle group, float **pview_buf, float **php_buf) {
    int buf[3];
    int height, width, n_channel;
    env_get_int_info(game, group, "view_size", buf);
    height = buf[0], width = buf[1], n_channel = buf[2];
    int n;
    env_get_num(game, group, &n);

    float *view_buf = new float[n * height * width * n_channel];
    float *hp_buf   = new float[n];

    float *bufs[] = {view_buf, hp_buf};
    env_get_observation(game, group, bufs);

    *pview_buf = view_buf;
    *php_buf   = hp_buf;
}

void get_position(EnvHandle game, GroupHandle group, int **ppos_x, int **ppos_y) {
    int n;
    env_get_num(game, group, &n);
    int *pos = new int [n * 2];
    int *pos_x = new int [n];
    int *pos_y = new int [n];
    env_get_int_info(game, group, "pos", pos);
    for (int i = 0; i < n; i++) {
        pos_x[i] = pos[i * 2];
        pos_y[i] = pos[i * 2 + 1];
    }
    *ppos_x = pos_x;
    *ppos_y = pos_y;
    delete [] pos;
}

void test_extract_view() {
    EnvHandle game;
    env_new_game(&game, "GridWorld");

    // config
    env_set_config(game, "map_width", 20);
    env_set_config(game, "map_height", 20);

    // register type
    const char *type_keys[] =  {
            "width", "length", "speed", "hp",
            "view_radius", "view_angle", "attack_radius", "attack_angle",
            "hear_radius", "speak_radius",
            "speak_ability", "damage", "trace", "step_recover", "kill_supply",
            "attack_in_group",
            "step_reward", "kill_reward", "dead_penalty",
    };

    float type_deer[] = {
            2, 2, 2, 10,
            3, 360, 0, 0,
            0, 0,
            0, 3, 0, -0.5f, 10,
            0,
            0, 0, -1,
    };

    float type_tiger[] = {
            1, 1, 2, 10,
            3, 120, 2, 120,
            0, 0,
            0, 3, 0, -0.5f, 10,
            0,
            0, 0, -1,
    };

    int n = sizeof(type_keys) / sizeof(type_keys[1]);
    gridworld_register_agent_type(game, "deer", n, type_keys, type_deer);
    gridworld_register_agent_type(game, "tiger", n, type_keys, type_tiger);

    // new group
    GroupHandle deer_handle, tiger_handle;
    gridworld_new_group(game, "deer", &deer_handle);
    gridworld_new_group(game, "tiger", &tiger_handle);

    env_reset(game);

    gridworld_add_agents(game, -1, 25, "random", nullptr, nullptr, nullptr);
    gridworld_add_agents(game, deer_handle, 20, "random", nullptr, nullptr, nullptr);
    gridworld_add_agents(game, tiger_handle, 10, "random", nullptr, nullptr, nullptr);
    env_set_render(game, "save_dir", "___debug___");
    env_render(game);

    int observation_handle = deer_handle;

    int buf[3];
    int t_height, t_width, t_n_channel;
    env_get_int_info(game, observation_handle, "view_size", buf);
    t_height = buf[0], t_width = buf[1], t_n_channel = buf[2];

    float (*view_buf)[t_height][t_width][t_n_channel];
    float *hp_buf;
    int *pos_x, *pos_y;

    env_get_num(game, observation_handle, &n);
    get_position(game, observation_handle, &pos_x, &pos_y);
    get_observation(game, observation_handle, (float **)&view_buf, &hp_buf);

    printf("observation\n");
    for (int i = 0; i < n; i++) {
        printf("======\n");
        printf("%d %d\n", pos_x[i], pos_y[i]);
        for (int j = 0; j < t_height; j++) {
            for (int k = 0; k < t_n_channel; k++) {
                for (int l = 0; l < t_width; l++) {
                    printf("%1.1f ", (view_buf[i][j][l][k]));
                }
                printf("  ");
            }
            printf("\n");
        }
    }

    printf("hp");
    for (int i = 0; i < n; i++) {
        printf("%.2f ", hp_buf[i]);
    }
}
#endif

int main() {
    //test_sector_range();
    //test_extract_view();

    return 0;
}
