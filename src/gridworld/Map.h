/**
 * \file Map.h
 * \brief The map for the game engine
 */

#ifndef MAGNET_GRIDWORLD_MAP_H
#define MAGNET_GRIDWORLD_MAP_H

#include <vector>
#include <random>
#include "grid_def.h"
#include "../Environment.h"
#include "Range.h"

namespace magent {
namespace gridworld {

typedef enum {BLANK, OBSTACLE} SlotType;
typedef enum {OCC_AGENT, OCC_FOOD} OccupyType;

typedef float Food;

class MapSlot {
public:
    MapSlot() : slot_type(BLANK), occupier(nullptr) {}
    SlotType slot_type;
    OccupyType occ_type;
    void *occupier;
};


class Map {
public:
    Map(): slots(nullptr), channel_ids(nullptr), w(-1), h(-1),
        wall_channel_id(0), food_channel_id(1) {
    }

    ~Map() {
        delete [] slots;
        delete [] channel_ids;
    }

    void reset(int width, int height, bool food_mode);

    Position get_random_blank(std::default_random_engine &random_engine, int width=1, int height=1);


    int add_agent(Agent *agent, Position pos, int width, int height, int base_channel_id);
    int add_agent(Agent *agent, int base_channel_id);

    int add_wall(Position pos);
    void remove_agent(Agent *agent);

    void average_pooling_group(float *group_buffer, int x0, int y0, int width, int height);
    void extract_view(const Agent *agent, float *linear_buffer, const int *channel_trans, const Range *range,
                      int n_channel, int width, int height, int view_x_offset, int view_y_offset,
                      int view_left_top_x, int view_left_top_y,
                      int view_right_bottom_x, int view_right_bottom_y) const;

    PositionInteger get_attack_obj(const AttackAction &attack, int &obj_x, int &obj_y) const;
    Reward do_attack(Agent *agent, PositionInteger pos_int, GroupHandle &dead_group);

    Reward do_move(Agent *agent, const int delta[2]);
    Reward do_turn(Agent *agent, int wise);

    int get_align(Agent *agent);

    void render();
    void get_wall(std::vector<Position> &walls) const;

private:
    MapSlot* slots;
    int *channel_ids;  // channel_id is supposed to be a member of MapSlot, extract it out from MapSlot for faster access of memory
    int w, h;
    const int wall_channel_id, food_channel_id;
    bool food_mode;

    /**
     * Utility
     */
    bool in_board(int x, int y) const {
        return x >= 0 && x < w && y >= 0 && y < h;
    }

    PositionInteger pos2int(Position pos) const {
        return pos2int(pos.x, pos.y);
    }

    PositionInteger pos2int(int x, int y) const {
        //return (PositionInteger)x * h + y;
        return (PositionInteger)y * w + x;
    }

    Position int2pos(PositionInteger pos) const {
        //return Position{(int)(pos / h), (int)(pos % h)};
        return Position{(int)(pos % w), (int)(pos / w)};
    }

    void set_channel_id(PositionInteger pos, int id) {
        channel_ids[pos] = id;
    }

    void dfs(std::default_random_engine &random_engine, int x, int y, int thick, int mode);

    inline bool is_blank_area(int x, int y, int width, int height,  void *self = nullptr);
    inline void clear_area(int x, int y, int width, int height);
    inline void fill_area(int x, int y, int width, int height,
                          void *occupier, OccupyType occ_type, int channel_id);
    inline void *get_collide(int x, int y, int width, int height, void *self);
};

} // namespace gridworld
} // namespace magent

#endif //MAGNET_GRIDWORLD_MAP_H
