/**
 * \file RenderGenerator.h
 * \brief Generate data for render
 */

#ifndef MAGENT_TRANSCITY_RENDER_H
#define MAGENT_TRANSCITY_RENDER_H

#include <string>

#include "city_def.h"
#include "Map.h"

namespace magent {
namespace trans_city {

class RenderGenerator {
public:
    RenderGenerator();

    // move to next file
    void next_file();

    void set_render(const char *key, const char *value);
    void gen_config(int w, int h);

    void render_a_frame(const std::vector<Agent *> &agents,
                        const std::vector<Position> &walls,
                        const std::vector<TrafficLight> &lights,
                        const std::vector<Park> &parks,
                        const std::vector<Building> &buildings);

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


#endif // MAGENT_TRANSCITY_RENDER_H
