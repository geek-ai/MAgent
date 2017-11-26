/**
 * \file Map.h
 * \brief The map for the game engine
 */

#ifndef MAGNET_DISCRETE_SNAKE_MAP_H
#define MAGNET_DISCRETE_SNAKE_MAP_H

#include <vector>
#include <set>
#include <assert.h>
#include "snake_def.h"

namespace magent {
namespace discrete_snake {

typedef enum {OCC_NONE, OCC_WALL, OCC_FOOD, OCC_AGENT} OccType;

struct Slot {
    OccType occ_type;
    void *occupier;
    int occ_ct;
};

class Map {
public:
    Map();
    ~Map();

    void reset(int wdith, int height);

    void add_agent(Agent *agent);
    int add_wall(Position pos);

    void get_wall(std::vector<Position> &walls) const;

    bool get_random_blank(std::vector<Position> &pos, int n);
    void extract_view(const Agent* agent, float *linear_buffer, int height, int width, int channel,
                      int id_counter);

    void move_tail(Agent *agent);
    void move_head(Agent *agent, PositionInteger head_int, Reward &reward, bool &dead, Food *&eaten);

    bool add_food(Food *food, int x, int y);
    void remove_food(Food *food);
    void make_food(Agent *agent, Reward value, std::vector<Food *> &foods, int add);
    int get_food_num();

    /**
     * Utility
     */
    bool in_board(int x, int y) const {
        return x >= 0 && x < map_width && y >= 0 && y < map_height;
    }

    PositionInteger pos2int(Position pos) const {
        return pos2int(pos.x, pos.y);
    }

    PositionInteger pos2int(int x, int y) const {
        return (PositionInteger)x * map_height + y;
        //return (PositionInteger)y * map_width + x;
    }

    Position int2pos(PositionInteger pos) const {
        return Position{(int)(pos / map_height), (int)(pos % map_height)};
        //return Position{(int)(pos % map_width), (int)(pos / map_width)};
    }

private:
    Slot *slots;

    int map_width, map_height;
};

} // namespace discrete_snake
} // namespace magent

#endif //MAGNET_DISCRETE_SNAKE_MAP_H
