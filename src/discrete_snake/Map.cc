/**
 * \file Map.cc
 * \brief The map for the game engine
 */

#include <random>
#include <stdexcept>
#include <assert.h>

#include "DiscreteSnake.h"
#include "Map.h"

namespace magent {
namespace discrete_snake {

Map::Map() : slots(nullptr) {
}

Map::~Map() {
    delete [] slots;
}

void Map::reset(int width, int height) {
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
        add_wall(Position{i, map_height - 1});
    }
    for (int i = 0; i < map_height; i++) {
        add_wall(Position{0, i});
        add_wall(Position{map_width - 1, i});
    }
}

int Map::add_wall(Position pos) {
    PositionInteger pos_int = pos2int(pos);
    if (slots[pos_int].occ_type != OCC_NONE)
        return 1;
    slots[pos_int].occ_type = OCC_WALL;
    return 0;
}

int Map::get_food_num() {
    int sum = 0;
    int size = map_width * map_height;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; i++)
        if (slots[i].occ_type == OCC_FOOD) {
            sum += 1;
        }
    return sum;
}

bool Map::get_random_blank(std::vector<Position> &pos, int n) {
    int tries = 0;

    pos.resize(n);
    while(tries < map_width * map_height) {
        int last_dir = 100;
        PositionInteger pos_int;

        int x = (int)random() % map_width;
        int y = (int)random() % map_height;

        int i;
        for (i = 0; i < n; i++) {
            pos_int = pos2int(x, y);
            if (slots[pos_int].occ_type != OCC_NONE)
                break;

            pos[i] = Position{x, y};

            int start = (int)random() % 100;
            for (int j = 0; j < 4; j++) { // 4 direction
                int new_x = x, new_y = y;
                int dir = (start + j) % 4;
                if (abs(dir - last_dir) == 2)
                    continue;
                switch (dir) {
                    case 0: new_x -= 1; break;
                    case 1: new_y -= 1; break;
                    case 2: new_x += 1; break;
                    case 3: new_y += 1; break;
                    default:
                        LOG(FATAL) << "invalid dir in Map::get_random_blank";
                }
                if (slots[pos_int].occ_type == OCC_NONE) {
                    x = new_x; y = new_y;
                    last_dir = dir;
                    break;
                }
            }
        }
        if (i == n) {
            return true;
        }
        tries++;
    }
    return false;
}

void Map::add_agent(Agent *agent) {
    for (auto pos : agent->get_body()) {
        PositionInteger pos_int = pos2int(pos);
        slots[pos_int].occ_type = OCC_AGENT;
        slots[pos_int].occupier = agent;
        slots[pos_int].occ_ct = 1;
    }
}

void Map::extract_view(const Agent* agent, float *linear_buffer, int height, int width, int channel,
                       int id_counter) {
    Position pos = agent->get_head();

    float (*buffer)[width][channel] = (decltype(buffer)) linear_buffer;

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
                    buffer[view_y][view_x][CHANNEL_WALL] = 1;
                    break;
                case OCC_FOOD:
                    buffer[view_y][view_x][CHANNEL_FOOD] = 1;
                    break;
                case OCC_AGENT:
                    occupier = (Agent*)slots[pos_int].occupier;
                    if (occupier == agent) {
                        buffer[view_y][view_x][CHANNEL_SELF] = 1;
                    } else {
                        buffer[view_y][view_x][CHANNEL_OTHER] = 1;
                    }
                    buffer[view_y][view_x][CHANNEL_ID] = (float) (occupier->get_id() + 1) / id_counter;
                    break;
            }
            view_y++;
        }
        view_x++;
    }
}

void Map::move_tail(Agent *agent) {
    Position tail = agent->pop_tail();
    PositionInteger tail_int = pos2int(tail);

    int remain = --slots[tail_int].occ_ct;

    if (remain == 0) {
        slots[tail_int].occ_type = OCC_NONE;
    }
}

void Map::move_head(Agent *agent, PositionInteger head_int, Reward &reward, bool &dead, Food *&eaten) {
    Food *food;
    int x, y, width, height; // for food

    switch (slots[head_int].occ_type) {
        case OCC_NONE:
            slots[head_int].occ_type = OCC_AGENT;
            slots[head_int].occupier = agent;
            slots[head_int].occ_ct = 1;
            reward = 0;
            dead = false;
            break;
        case OCC_AGENT:
            if (slots[head_int].occupier != agent) {
                Agent *other = (Agent *)slots[head_int].occupier;
                dead = true;
            } else {
                slots[head_int].occ_ct++;
            }
            break;
        case OCC_WALL:
            dead = true;
            break;
        case OCC_FOOD:
            food = (Food *) slots[head_int].occupier;
            eaten = food;
            int x, y;
            food->get_xy(x, y);

            reward = food->get_value();

            slots[head_int].occ_type = OCC_AGENT;
            slots[head_int].occupier = agent;
            slots[head_int].occ_ct = 1;
            break;
    }
}

void Map::make_food(Agent *agent, Reward value, std::vector<Food*> &foods, int add) {
    bool skip_head = true;
    int ct = 0;
    for (Position pos : agent->get_body()) {
        if (skip_head) {
            skip_head = false;
        } else {
            PositionInteger pos_int = pos2int(pos);

            if (slots[pos_int].occ_type == OCC_AGENT) {
                if (ct < add) {
                    Food *food = new Food(1, 1, value);
                    slots[pos_int].occ_type = OCC_FOOD;
                    slots[pos_int].occupier = food;
                    food->set_xy(pos.x, pos.y);
                    foods.push_back(food);
                    ct++;
                } else {
                    slots[pos_int].occ_type = OCC_NONE;
                }
            }
        }
    }
}

bool Map::add_food(Food *food, int x, int y) {
    int width, height;
    PositionInteger  pos_int;
    food->set_xy(x, y);
    food->get_size(width, height);

    // check clean
    for (int i = 0; i < width; i++)
        for (int j = 0; j < height; j++) {
            pos_int = pos2int(x + i, y + j);
            if (slots[pos_int].occ_type != OCC_NONE)
                return false;
        }

    // mark food
    for (int i = 0; i < width; i++)
        for (int j = 0; j < height; j++) {
            pos_int = pos2int(x + i, y + j);
            slots[pos_int].occ_type = OCC_FOOD;
            slots[pos_int].occupier = food;
        }
    return true;
}

void Map::remove_food(Food *food) {
    int x, y, width, height;
    food->get_xy(x, y);
    food->get_size(width, height);
    // clear food
    for (int i = 0; i < width; i++)
        for (int j = 0; j < height; j++) {
            PositionInteger pos_int = pos2int(x + i, y + j);
            if (slots[pos_int].occ_type == OCC_FOOD) {
                slots[pos_int].occ_type = OCC_NONE;
                slots[pos_int].occupier = nullptr;
            }
        }
}

void Map::get_wall(std::vector<Position> &walls) const {
    for (int i = 0; i < map_width * map_height; i++) {
        if (slots[i].occ_type == OCC_WALL)
            walls.push_back(int2pos(i));
    }
}

} // namespace discrete_snake
} // namespace magent