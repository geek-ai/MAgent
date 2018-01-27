/**
 * \file Map.h
 * \brief The map for the game engine
 */


#ifndef MAGENT_TRANS_CITY_MAP_H
#define MAGENT_TRANS_CITY_MAP_H

#include <vector>
#include <set>
#include <map>
#include <assert.h>
#include <random>
#include "city_def.h"

namespace magent {
namespace trans_city {

typedef enum {OCC_NONE, OCC_WALL, OCC_LIGHT, OCC_PARK, OCC_AGENT} OccType;

struct Slot {
    OccType occ_type;
    void *occupier;
    int occ_ct;
};

class Map {
public:
    Map();
    ~Map();

    void reset(std::vector<Position> &walls, int width, int height);

    int add_agent(Agent* agent);
    int add_wall(Position pos);
    int add_light(Position pos, int w, int h);
    int add_park(Position pos, int w, int h, int no);

    Position get_random_blank(std::default_random_engine &random_engine);
    void extract_view(const Agent* agent, float *linear_buffer, int height, int width, int channel);

    int do_move(Agent *agent, const int *delta,
                const std::map<std::pair<Position, Position>, TrafficLine> &lines,
                const std::vector<TrafficLight> &lights);

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

} // namespace trans_city
} // namespace magent

#endif //MAGENT_TRANS_CITY_MAP_H
