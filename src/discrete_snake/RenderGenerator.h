/**
 * \file RenderGenerator.h
 * \brief Generate data for render
 */

#ifndef MAGNET_DISCRETE_SNAKE_RENDER_H
#define MAGNET_DISCRETE_SNAKE_RENDER_H

#include <string>

#include "snake_def.h"
#include "Map.h"

namespace magent {
namespace discrete_snake {

class RenderGenerator {
public:
    RenderGenerator();

    void next_file();

    void set_render(const char *key, const char *value);
    void gen_config(const Map &map, int w, int h);

    void render_a_frame(const std::vector<Agent *> &agents, const std::set<Food *> &foods);


    std::string get_save_dir() {
        return save_dir;
    }

private:
    std::string save_dir;

    int file_ct;
    int frame_ct;
    int frame_per_file;
    unsigned int id_ct;
};

} // namespace magent
} // namespace discrete_snake

#endif //MAGNET_DISCRETE_SNAKE_RENDER_H
