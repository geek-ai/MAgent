/**
 * \file RenderGenerator.h
 * \brief Generate data for render
 */

#ifndef MAGNET_GRIDWORLD_RENDER_H
#define MAGNET_GRIDWORLD_RENDER_H

#include <vector>
#include <string>

#include "grid_def.h"
#include "Map.h"

namespace magent {
namespace gridworld {

struct RenderAttackEvent{
    int id;
    int x, y;
};

class RenderGenerator {
public:
    RenderGenerator();

    // move to next file
    void next_file();

    // getter and setter for attack_events
    void set_attack_event(std::vector<RenderAttackEvent> &attack_events);
    std::vector<RenderAttackEvent> &get_attack_event();

    void set_render(const char *key, const char *value);
    void gen_config(std::vector<Group> &group, int w, int h);

    void render_a_frame(std::vector<Group> &groups, const Map &map);

    std::string get_save_dir() {
        return save_dir;
    }

private:
    std::string save_dir;

    int file_ct;
    int frame_ct;
    int frame_per_file;

    std::vector<RenderAttackEvent> attack_events;
};

} // namespace gridworld
} // namespace magent

#endif //MAGNET_GRIDWORLD_RENDER_H
