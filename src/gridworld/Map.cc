/**
 * \file Map.cc
 * \brief The map for the game engine
 */

#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include "Map.h"
#include "GridWorld.h"

namespace magent {
namespace gridworld {

inline void abs_to_rela(int c_x, int c_y, Direction dir, int abs_x, int abs_y, int &rela_x, int &rela_y);
inline void rela_to_abs(int c_x, int c_y, Direction dir, int rela_x, int rela_y, int &abs_x, int &abs_y);
inline void save_to_real(const Agent *agent, int &real_x, int &real_y);
inline void real_to_save(const Agent *agent, int real_x, int real_y, Direction new_dir, int &save_x, int &save_y);
inline void get_size_for_dir(Agent *agent, int &width, int &height);

#define MAP_INNER_Y_ADD w

void Map::reset(int width, int height, bool food_mode) {
    this->w = width;
    this->h = height;
    this->food_mode = food_mode;

    if (slots != nullptr)
        delete [] slots;
    slots = new MapSlot[w * h];

    if (channel_ids != nullptr)
        delete [] channel_ids;
    channel_ids = new int[w * h];

    memset(channel_ids, -1, sizeof(int) * w * h);

    // init border
    for (int i = 0; i < w; i++) {
        add_wall(Position{i, 0});
        add_wall(Position{i, h-1});
    }
    for (int i = 0; i < h; i++) {
        add_wall(Position{0, i});
        add_wall(Position{w-1, i});
    }
}

Position Map::get_random_blank(std::default_random_engine &random_engine, int width, int height) {
    int tries = 0;
    while (true) {
        int x = (int) random_engine() % (w - width);
        int y = (int) random_engine() % (h - height);

        if (is_blank_area(x, y, width, height)) {
            return Position{x, y};
        }

        if (tries++ > w * h) {
            LOG(FATAL) << "cannot find a blank position in a filled map";
        }
    }
}

int Map::add_agent(Agent *agent, Position pos, int width, int height, int base_channel_id) {
    if (is_blank_area(pos.x, pos.y, width, height)) {
        // fill in map
        fill_area(pos.x, pos.y, width, height, agent, OCC_AGENT, base_channel_id);
        return 0;
    } else {
        return 1;
    }
}

int Map::add_agent(Agent *agent, int base_channel_id) {
    Direction dir = agent->get_dir();
    Position  pos = agent->get_pos();
    int width = agent->get_type().width, length = agent->get_type().length;

    int m_width, m_height;

    if (dir == NORTH || dir == SOUTH) {
        m_width = width;
        m_height = length;
    } else {
        m_width = length;
        m_height = width;
    }

    if (is_blank_area(pos.x, pos.y, m_width, m_height)) {
        // fill in map
        fill_area(pos.x, pos.y, m_width, m_height, agent, OCC_AGENT, base_channel_id);
        return 0;
    } else {
        return 1;
    }
}

void Map::remove_agent(Agent *agent) {
    Position pos = agent->get_pos();
    int width, height;
    get_size_for_dir(agent, width, height);

    // clear map
    clear_area(pos.x, pos.y, width, height);
}

int Map::add_wall(Position pos) {
    PositionInteger pos_int = pos2int(pos);
    if (slots[pos_int].slot_type == BLANK && slots[pos_int].occupier != nullptr)
        return 1;
    slots[pos_int].slot_type = OBSTACLE;
    set_channel_id(pos_int, wall_channel_id);
    return 0;
}

void Map::average_pooling_group(float *group_buffer, int x0, int y0, int width, int height) {
    for (int x = x0; x < x0 + width; x++) {
        for (int y = y0; y < y0 + height; y++) {
            PositionInteger pos_int = pos2int(x, y);
            if (slots[pos_int].occupier != nullptr && slots[pos_int].occ_type == OCC_AGENT) {
                Agent *agent = (Agent *)slots[pos_int].occupier;
                group_buffer[agent->get_group()]++;
            }
        }
    }
}

void Map::extract_view(const Agent *agent, float *linear_buffer, const int *channel_trans, const Range *range,
                       int n_channel, int width, int height, int view_x_offset, int view_y_offset,
                       int view_left_top_x, int view_left_top_y,
                       int view_right_bottom_x, int view_right_bottom_y) const {
    // convert coordinates between absolute map and relative view
    Direction dir = agent->get_dir();

    int agent_x, agent_y;
    int eye_x, eye_y;
    int x1, y1, x2, y2;

    save_to_real(agent, agent_x, agent_y);
    rela_to_abs(agent_x, agent_y, dir, view_x_offset, view_y_offset, eye_x, eye_y);
    rela_to_abs(eye_x, eye_y, dir, view_left_top_x, view_left_top_y, x1, y1);
    rela_to_abs(eye_x, eye_y, dir, view_right_bottom_x, view_right_bottom_y, x2, y2);

    // find the coordinate of start point and end point in map
    int start_x, start_y, end_x, end_y;
    start_x = std::max(std::min(x1, x2), 0);
    end_x = std::min(std::max(x1, x2), w - 1);
    start_y = std::max(std::min(y1, y2), 0);
    end_y = std::min(std::max(y1, y2), h - 1);

    NDPointer<float, 3> buffer(linear_buffer, {height, width, n_channel});

    // build projection from map coordinate to view buffer coordinate
    int view_x, view_y;
    int view_rela_x, view_rela_y;
    abs_to_rela(eye_x, eye_y, dir, start_x, start_y, view_rela_x, view_rela_y);
    view_x = view_rela_x - view_left_top_x;
    view_y = view_rela_y - view_left_top_y;

    int *p_view_inner, *p_view_outer;
    int d_view_inner, d_view_outer;
    switch (dir) {
        case NORTH:
            p_view_inner = &view_y; p_view_outer = &view_x;
            d_view_inner = 1; d_view_outer = 1;
            break;
        case SOUTH:
            p_view_inner = &view_y; p_view_outer = &view_x;
            d_view_inner = -1; d_view_outer = -1;
            break;
        case EAST:
            p_view_inner = &view_x; p_view_outer = &view_y;
            d_view_inner = 1; d_view_outer = -1;
            break;
        case WEST:
            p_view_inner = &view_x; p_view_outer = &view_y;
            d_view_inner = -1; d_view_outer = 1;
            break;
        default:
            LOG(FATAL) << "invalid direction in Map::extract_view";
    }

    int start_inner = *p_view_inner;

    // scan the map
    for (int x = start_x; x <= end_x; x++) {
        PositionInteger pos_int = pos2int(x, start_y);
        for (int y = start_y; y <= end_y; y++) {
            int channel_id = channel_ids[pos_int];

            if (channel_id != -1 && range->is_in(view_y, view_x)) {
                channel_id = channel_trans[channel_id];
                buffer.at(view_y, view_x, channel_id) = 1;
                if (slots[pos_int].occupier != nullptr && slots[pos_int].occ_type == OCC_AGENT) { // is agent
                    Agent *p = ((Agent *) slots[pos_int].occupier);
                    buffer.at(view_y, view_x, channel_id + 1) = p->get_hp() / p->get_type().hp; // normalize hp
                }
            }

            *p_view_inner += d_view_inner;
            pos_int += MAP_INNER_Y_ADD;
        }
        *p_view_inner = start_inner;
        *p_view_outer += d_view_outer;
    }
}

PositionInteger Map::get_attack_obj(const AttackAction &attack, int &obj_x, int &obj_y) const {
    const Agent *agent = attack.agent;
    const AgentType *type = &attack.agent->get_type();
    Direction dir = agent->get_dir();

    int att_x_offset = type->att_x_offset, att_y_offset = type->att_y_offset;
    int agent_x, agent_y;
    int rela_x, rela_y;

    agent->get_type().attack_range->num2delta(attack.action, rela_x, rela_y);

    save_to_real(agent, agent_x, agent_y);
    rela_to_abs(agent_x, agent_y, dir, att_x_offset + rela_x, att_y_offset + rela_y, obj_x, obj_y);

    if (!in_board(obj_x, obj_y)) {
        return -1;
    }

    PositionInteger pos_int = pos2int(obj_x, obj_y);

    if (slots[pos_int].occupier == nullptr) {
        return -1;
    }

    switch (slots[pos_int].occ_type) {
        case OCC_AGENT:
        {
            Agent *obj = (Agent *) slots[pos_int].occupier;

            if (!type->attack_in_group && agent->get_group() == obj->get_group()) { // same type
                return -1;
            } else {
                return pos_int;
            }
            break;
        }
        case OCC_FOOD:
            return pos_int;
        default:
            LOG(FATAL) << "invalid occ_type in Map::get_attack_obj";

    }
    return -1;
}

// do attack for agent, return kill_reward and dead_group
Reward Map::do_attack(Agent *agent, PositionInteger pos_int, GroupHandle &dead_group) {
    // !! all the check should be done at Map::get_attack_obj

    if (slots[pos_int].occupier == nullptr)  // dead
        return 0.0;

    switch(slots[pos_int].occ_type) {
        case OCC_AGENT:
        {
            Agent *obj = ((Agent *)slots[pos_int].occupier);

            obj->be_attack(agent->get_type().damage);
            if (obj->is_dead()) {
                agent->set_last_op(OP_KILL);
                agent->set_op_obj(obj);

                // remove dead people
                remove_agent(obj);
                dead_group = obj->get_group();
                agent->add_hp(obj->get_type().kill_supply);

                // add food
                if (food_mode) {
                    slots[pos_int].occ_type = OCC_FOOD;
                    Food *food = new Food;
                    *food = obj->get_type().food_supply;
                    slots[pos_int].occupier = food;
                    set_channel_id(pos_int, food_channel_id);
                }
                return obj->get_type().kill_reward;
            } else {
                agent->set_last_op(OP_ATTACK);
                agent->set_op_obj(obj);
                return 0.0;
            }
            break;
        }
        case OCC_FOOD:  // deprecated ?
        {
            Food *food = (Food *)slots[pos_int].occupier;
            float add = std::min(agent->get_type().eat_ability, *food);
            agent->add_hp(add);
            *food -= add;
            if (*food < 0.1) {
                slots[pos_int].occupier = nullptr;
                set_channel_id(pos_int, -1);
                delete food;
            }
            break;
        }
        default:
            LOG(FATAL) << "invalid occ_type in Map::do_attack";
    }

    return 0.0;
}

// do move for agent, dx = delta[0], dy = delta[1]
Reward Map::do_move(Agent *agent, const int delta[2]) {
    Position &pos = agent->get_pos();
    const int new_x = pos.x + delta[0], new_y = pos.y + delta[1];
    int width, height;
    get_size_for_dir(agent, width, height);

    bool blank = is_blank_area(new_x, new_y, width, height, agent);
    if (blank) {
        PositionInteger old_pos_int = pos2int(pos);

        // backup old
        void *occupier = slots[old_pos_int].occupier;
        OccupyType occ_type = slots[old_pos_int].occ_type;
        int channel_id  = channel_ids[old_pos_int];

        clear_area(pos.x, pos.y, width, height);
        fill_area(new_x, new_y, width, height, occupier, occ_type, channel_id);

        pos.x = new_x;
        pos.y = new_y;
    } else {
        void *collide = get_collide(new_x, new_y, width, height, agent);
        if  (collide != nullptr) {
            /*Agent *obj = (Agent *)collide;
            if (agent->get_group() != obj->get_group())
                printf("%d %d\n", agent->get_group(), obj->get_group());*/

            Agent *obj = (Agent *)collide;
            if (obj->get_type().can_absorb) { // special condition
                if (!obj->is_absorbed()) {
                    obj->set_absorbed(true);
                    obj->set_hp(obj->get_hp() * 2);
                    agent->set_dead(true);
                    remove_agent(agent);
                    agent->set_last_op(OP_COLLIDE);
                    agent->set_op_obj(collide);
                }
            } else {
                agent->set_last_op(OP_COLLIDE);
                agent->set_op_obj(collide);
            }
        } else {
        }
    }
    return 0.0;
}

// to turn, wise = 1 for clockwise, wise = -1 for counter-clockwise
Reward Map::do_turn(Agent *agent, int wise) {
    int width, height;

    Position &pos = agent->get_pos();
    PositionInteger pos_int = pos2int(pos);
    Direction dir = agent->get_dir();
    Direction new_dir = (Direction) ((dir + wise + DIR_NUM) % DIR_NUM);

    get_size_for_dir(agent, width, height);

    int agent_x, agent_y;
    int anchor_x, anchor_y;
    int new_x, new_y;
    int save_x, save_y;
    save_to_real(agent, agent_x, agent_y);
    rela_to_abs(agent_x, agent_y, dir, agent->get_type().turn_x_offset, agent->get_type().turn_y_offset,
                anchor_x, anchor_y);

    int dx = agent_x - anchor_x, dy = agent_y - anchor_y;
    if (wise == -1) { // wise = 1 for clockwise, = -1 for counter-clockwise
        new_x = anchor_x - dy;
        new_y = anchor_y + dx;
    } else{
        new_x = anchor_x + dy;
        new_y = anchor_y - dx;
    }

    real_to_save(agent, new_x, new_y, new_dir, save_x, save_y);

    if (is_blank_area(save_x, save_y, height, width, agent)) {
        // backup old
        void *occupier = slots[pos_int].occupier;
        OccupyType occ_type = slots[pos_int].occ_type;
        int channel_id  = channel_ids[pos_int];

        // clear old
        clear_area(pos.x, pos.y, width, height);

        // fill new
        agent->set_dir(new_dir);

        fill_area(save_x, save_y, height, width, occupier, occ_type, channel_id);
        pos.x = save_x; pos.y = save_y;
    }
    return 0.0;
}

int Map::get_align(Agent *agent) {
    Position pos = agent->get_pos();
    GroupHandle group = agent->get_group();
    PositionInteger max_size = w * h;

    // scan x axis
    PositionInteger pos_int = pos2int(pos);
    int x_align = -1;
    do {  // positive direction
        x_align++;
        pos_int++;
    } while (slots[pos_int].occupier != nullptr &&   // NOTE: do not check boundary, since the map has walls
        ((Agent *)slots[pos_int].occupier)->get_group() == group);

    pos_int = pos2int(pos);
    do {  // negtive direction
        x_align++;
        pos_int--;
    } while (slots[pos_int].occupier != nullptr &&
        ((Agent *)slots[pos_int].occupier)->get_group() == group);

    // scan y axis
    pos_int = pos2int(pos);
    int y_align = -1;
    do {  // positive direction
        y_align++;
        pos_int += MAP_INNER_Y_ADD;
    } while (slots[pos_int].occupier != nullptr &&
        ((Agent *)slots[pos_int].occupier)->get_group() == group);

    pos_int = pos2int(pos);
    do {  // negtive direction
        y_align++;
        pos_int -= MAP_INNER_Y_ADD;
    } while (slots[pos_int].occupier != nullptr &&
        ((Agent *)slots[pos_int].occupier)->get_group() == group);

    return std::max(x_align, y_align);
}

/**
 * Utility to operate map
 */

// check if rectangle (x,y) - (x + width, y + height) is a blank area
// the rectangle can only contains blank slots and itself
inline bool Map::is_blank_area(int x, int y, int width, int height, void *self) {
    if (x < 0 || y < 0 || x + width >= w || y + height >= h)
        return false;
    bool blank = true;
    for (int i = 0; i < width && blank; i++) {
        PositionInteger pos_int = pos2int(x + i, y);
        for (int j = 0; j < height && blank; j++) {
            if (slots[pos_int].slot_type != BLANK ||
                (slots[pos_int].occupier != nullptr &&
                    slots[pos_int].occupier != self)) {
                blank = false;
            }
            pos_int += MAP_INNER_Y_ADD;
        }
    }
    return blank;
}

// fill a rectangle (x, y) - (x + width, y + height) with specific occupier
inline void Map::fill_area(int x, int y, int width, int height, void *occupier, OccupyType occ_type, int channel_id) {
    for (int i = 0; i < width; i++) {
        PositionInteger pos_int = pos2int(x + i, y);
        for (int j = 0; j < height; j++) {
            slots[pos_int].occupier = occupier;
            slots[pos_int].occ_type = occ_type;
            set_channel_id(pos_int, channel_id);
            pos_int += MAP_INNER_Y_ADD;
        }
    }
}

// get original occupier in the rectangle who results in a collide with a move intention
inline void * Map::get_collide(int x, int y, int width, int height, void *self) {
    if (x < 0 || y < 0 || x + width >= w || y + height >= h)
        return nullptr;
    void *collide_obj = nullptr;
    for (int i = 0; i < width && collide_obj == nullptr; i++) {
        PositionInteger pos_int = pos2int(x + i, y);
        for (int j = 0; j < height && collide_obj == nullptr; j++) {
            if (slots[pos_int].occ_type == OCC_AGENT &&
                slots[pos_int].occupier != self) {
                collide_obj = slots[pos_int].occupier;
            }
            pos_int += MAP_INNER_Y_ADD;
        }
    }
    return collide_obj;
}

// clear a rectangle
inline void Map::clear_area(int x, int y, int width, int height) {
    for (int i = 0; i < width; i++) {
        PositionInteger pos_int = pos2int(x + i, y);
        for (int j = 0; j < height; j++) {
            slots[pos_int].occupier = nullptr;
            set_channel_id(pos_int, -1);
            pos_int += MAP_INNER_Y_ADD;
        }
    }
}

inline void rela_to_abs(int c_x, int c_y, Direction dir, int rela_x, int rela_y, int &abs_x, int &abs_y) {
    switch (dir) {
        case NORTH:
            abs_x = c_x + rela_x; abs_y = c_y + rela_y;
            break;
        case SOUTH:
            abs_x = c_x - rela_x; abs_y = c_y - rela_y;
            break;
        case WEST:
            abs_x = c_x + rela_y; abs_y = c_y - rela_x;
            break;
        case EAST:
            abs_x = c_x - rela_y; abs_y = c_y + rela_x;
            break;
        default:
            LOG(FATAL) << "invalid direction in rela_to_abs";
    }
}

inline void abs_to_rela(int c_x, int c_y, Direction dir, int abs_x, int abs_y, int &rela_x, int &rela_y) {
    switch (dir) {
        case NORTH:
            rela_x = abs_x - c_x; rela_y = abs_y - c_y;
            break;
        case SOUTH:
            rela_x = c_x - abs_x; rela_y = c_y - abs_y;
            break;
        case WEST:
            rela_y = abs_x - c_x; rela_x = c_y - abs_y;
            break;
        case EAST:
            rela_y = c_x - abs_x; rela_x = abs_y - c_y;
            break;
        default:
            LOG(FATAL) << "invalid direction in abs_to_rela";
    }
}

inline void save_to_real(const Agent *agent, int &real_x, int &real_y) {
    Direction dir = agent->get_dir();
    Position pos = agent->get_pos();
    int width = agent->get_type().width, length = agent->get_type().length;

    switch(dir) {
        case NORTH:
            real_x = pos.x; real_y = pos.y;
            break;
        case SOUTH:
            real_x = pos.x + width - 1; real_y = pos.y + length - 1;
            break;
        case WEST:
            real_x = pos.x; real_y = pos.y + width - 1;
            break;
        case EAST:
            real_x = pos.x + length - 1; real_y = pos.y;
            break;
        default:
            LOG(FATAL) << "invalid direction in save_to_real";
    }
}

inline void real_to_save(const Agent *agent, int real_x, int real_y, Direction new_dir, int &save_x, int &save_y) {
    int width = agent->get_type().width, length = agent->get_type().length;

    switch(new_dir) {
        case NORTH:
            save_x = real_x; save_y = real_y;
            break;
        case SOUTH:
            save_x = real_x - width + 1; save_y = real_y - length + 1;
            break;
        case WEST:
            save_x = real_x; save_y = real_y - width + 1;
            break;
        case EAST:
            save_x = real_x - length + 1; save_y = real_y;
            break;
        default:
            LOG(FATAL) << "invalid direction in real_to_save";
    }
}

inline void get_size_for_dir(Agent *agent, int &width, int &height) {
    Direction dir = agent->get_dir();

    if (dir == NORTH || dir == SOUTH) {
        width = agent->get_type().width;
        height = agent->get_type().length;
    } else {
        width = agent->get_type().length;
        height = agent->get_type().width;
    }
}

void Map::get_wall(std::vector<Position> &walls) const {
    for (int i = 0; i < w * h; i++) {
        if (slots[i].slot_type == OBSTACLE) {
            walls.push_back(int2pos(i));
        }
    }
}

/**
 * Render for debug, print the map to terminal screen
 */
void Map::render() {
    for (int x = 0; x < w; x++)
        printf("=");        puts("");
    printf("    ");
    for (int x = 0; x < w; x++)
        printf("%2d ", x);  puts("");

    for (int y = 0; y < h; y++) {
        printf("%2d ", y);
        for (int x = 0; x < w; x++) {
            MapSlot s = slots[pos2int(x, y)];
            char buf[4] = {0, 0, 0};

            switch (s.slot_type) {
                case BLANK:
                    if (s.occupier == nullptr) {
                        buf[0] = ' ';
                    } else {
                        switch (s.occ_type) {
                            case OCC_AGENT: {
                                Agent &agent = *((Agent *)s.occupier);
                                switch (agent.get_dir()) {
                                    case EAST:  buf[0] = '>'; break;
                                    case WEST:  buf[0] = '<'; break;
                                    case NORTH: buf[0] = '^'; break;
                                    case SOUTH: buf[0] = 'v'; break;
                                    default:
                                        LOG(FATAL) << "invalid direction in Map::render";
                                }
                                buf[1] = (char)toupper(agent.get_type().name[0]);
                            }
                                break;
                            case OCC_FOOD:
                                buf[0] = '+';
                                break;
                            default:
                                LOG(FATAL) << "invalid occ type in Map::render";
                        }
                    }
                    break;
                case OBSTACLE:
                    buf[0] = '#';
                    break;
                default:
                    LOG(FATAL) << "invalid slot type in Map::render";

            }
            printf("%3s", buf);
        }
        printf("\n");
    }

    for (int x = 0; x < w; x++)
        printf("=");        puts("\n");
}

} // namespace magent
} // namespace gridworld