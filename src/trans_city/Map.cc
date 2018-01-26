/**
 * \file Map.cc
 * \brief The map for the game engine
 */

#include <random>
#include "TransCity.h"
#include "Map.h"

namespace magent {
namespace trans_city {

Map::Map() : slots(nullptr) {
}

Map::~Map() {
    delete [] slots;
}

void Map::reset(std::vector<Position> &walls, int width, int height) {
    if (slots != nullptr)
        delete [] slots;
    slots = new Slot[width * height];

    for (int i = 0; i < width * height; i++) {
        slots[i].occ_type = OCC_NONE;
    }

    map_width = width;
    map_height = height;

    // init border
    for (int i = 0; i < map_width; i++) {
        add_wall(Position{i, 0});
        add_wall(Position{i, map_height-1});
        walls.push_back(Position{i, 0});
        walls.push_back(Position{i, map_height-1});
    }
    for (int i = 0; i < map_height; i++) {
        add_wall(Position{0, i});
        add_wall(Position{map_width-1, i});
        walls.push_back(Position{0, i});
        walls.push_back(Position{map_height-1, i});
    }
}

Position Map::get_random_blank(std::default_random_engine &random_engine) {
    int tries = 0;
    while (true) {
        int x = (int) random_engine() % (map_width);
        int y = (int) random_engine() % (map_height);

        if (slots[pos2int(x, y)].occ_type == OCC_NONE) {
            return Position{x, y};
        }

        if (tries++ > map_width * map_height) {
            LOG(FATAL) << "cannot find a blank position in a filled map";
        }
    }
}

int Map::add_agent(Agent *agent) {
    PositionInteger pos_int = pos2int(agent->get_pos());
    if (slots[pos_int].occ_type != OCC_NONE)
        return 0;
    slots[pos_int].occ_type = OCC_AGENT;
    slots[pos_int].occupier = agent;
    slots[pos_int].occ_ct = 1;
    return 1;
}

int Map::add_wall(Position pos) {
    PositionInteger pos_int = pos2int(pos);
    if (slots[pos_int].occ_type != OCC_NONE)
        return 0;
    slots[pos_int].occ_type = OCC_WALL;
    return 1;
}

int Map::add_light(Position pos, int w, int h) {
    int x = pos.x;
    int y = pos.y;

    Position poss[] = {
            Position{x, y},
            Position{x, y+h},
            Position{x+w, y},
            Position{x+w, y+h}
    };

    for (auto pos : poss) {
        if (slots[pos2int(pos)].occ_type != OCC_NONE && slots[pos2int(pos)].occ_type != OCC_WALL)
            return 0;
    }

    slots[pos2int(x, y)].occ_type = OCC_LIGHT;
    slots[pos2int(x, y+h)].occ_type = OCC_LIGHT;
    slots[pos2int(x+w, y)].occ_type = OCC_LIGHT;
    slots[pos2int(x+w, y+h)].occ_type = OCC_LIGHT;

    return 1;
}

int Map::add_park(Position pos) {
    PositionInteger pos_int = pos2int(pos);
    if (slots[pos_int].occ_type != OCC_NONE)
        return 0;
    slots[pos_int].occ_type = OCC_PARK;
    return 1;
}

void Map::extract_view(const Agent* agent, float *linear_buffer, int height, int width, int channel) {
    Position pos = agent->get_pos();

    NDPointer<float, 3> buffer(linear_buffer, {{height, width, channel}});

    int x_start = pos.x - width / 2;
    int y_start = pos.y - height / 2;
    int x_end = x_start + width - 1;
    int y_end = y_start + height - 1;

    x_start = std::max(0, std::min(map_width-1, x_start));
    x_end   = std::max(0, std::min(map_width-1, x_end));
    y_start = std::max(0, std::min(map_height-1, y_start));
    y_end   = std::max(0, std::min(map_height-1, y_end));

    int view_x_start = 0 + x_start - (pos.x - width/2);
    int view_y_start = 0 + y_start - (pos.y - height/2);

    int view_x = view_x_start;
    for (int x = x_start; x <= x_end; x++) {
        int view_y = view_y_start;
        for (int y =  y_start; y <= y_end; y++) {
            PositionInteger pos_int = pos2int(x, y);
            Agent *occupier;
            switch (slots[pos_int].occ_type) {
                case OCC_NONE:
                    break;
                case OCC_WALL:
                    buffer.at(view_y, view_x, CHANNEL_WALL) = 1;
                    break;
                case OCC_PARK:
                    buffer.at(view_y, view_x, CHANNEL_PARK) = 1;
                    break;
                case OCC_LIGHT:
                    buffer.at(view_y, view_x, CHANNEL_LIGHT) = 1;
                    break;
                case OCC_AGENT:
                    occupier = (Agent*)slots[pos_int].occupier;
                    if (occupier == agent) {
                        buffer.at(view_y, view_x, CHANNEL_SELF) = 1;
                    } else {
                        buffer.at(view_y, view_x, CHANNEL_OTHER) = 1;
                    }
                    break;
            }
            view_y++;
        }
        view_x++;
    }
}

int Map::do_move(Agent *agent, const int *delta,
                 const std::map<std::pair<Position, Position>, TrafficLine> &lines,
                 const std::vector<TrafficLight> &lights) {
    Position pos = agent->get_pos();
    Position new_pos = pos;
    new_pos.x += delta[0];
    new_pos.y += delta[1];


    if (slots[pos2int(new_pos)].occ_type == OCC_NONE) {
        auto iter = lines.find(std::make_pair(pos, new_pos));
        if (iter != lines.end() && !iter->second.can_pass(lights))
            return 0;

        slots[pos2int(new_pos)].occ_type = OCC_AGENT;
        slots[pos2int(new_pos)].occupier = agent;
        slots[pos2int(pos)].occ_type = OCC_NONE;
        agent->set_pos(new_pos);
        return 1;
    } else {
        return 0;
    }
}


} // namespace trans_city
} // namespace magent
