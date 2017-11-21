/**
 * \file RenderGenerator.cc
 * \brief Generate data for render
 */

#include <ios>
#include <fstream>
#include <string>
#include <sstream>

#include "RenderGenerator.h"
#include "DiscreteSnake.h"
#include "../utility/utility.h"

namespace magent {
namespace discrete_snake {

RenderGenerator::RenderGenerator() {
    save_dir = "";
    file_ct = frame_ct = 0;
    frame_per_file = 10000;
    id_ct = 0;
}

void RenderGenerator::next_file() {
    file_ct++;
    frame_ct = 0;
}

void RenderGenerator::set_render(const char *key, const char *value) {
    if (strequ(key, "save_dir"))
        save_dir = std::string(value);
    else if (strequ(key, "frame_per_file"))
        sscanf(value, "%d", &frame_per_file);
}

template <typename T>
void print_json(std::ofstream &os, const char *key, T value, bool last=false) {
    os << "\"" << key << "\": " << value;
    if (last)
        os << std::endl;
    else
        os << "," << std::endl;
}

std::string rgba_string(int r, int g, int b, float alpha) {
    std::stringstream ss;
    ss << "\"rgba(" << r << "," << g << "," << b << "," << alpha << ")\"";
    return ss.str();
};

void RenderGenerator::gen_config(const Map &map, int w, int h) {
    /***** config *****/
    std::ofstream f_config(save_dir + "/" + "config.json");

    int colors[][3] = {
        {192, 64, 64},
        {64, 64, 192},
        {64, 192, 64},
    };
    int i;

    f_config << "{" << std::endl;
    print_json(f_config, "width", w);
    print_json(f_config, "height", h);
    print_json(f_config, "static-file", "\"static.map\"");
    print_json(f_config, "obstacle-style", rgba_string(127, 127, 127, 1));
    print_json(f_config, "dynamic-file-directory", "\".\"");
    print_json(f_config, "attack-style", rgba_string(63, 63, 63, 0.8));
    print_json(f_config, "minimap-width", 300);
    print_json(f_config, "minimap-height", 250);

    // groups
    f_config << "\"group\" : [" << std::endl;

    // food
    i = 1;
    f_config << "{" << std::endl;
    print_json(f_config, "height", 1);
    print_json(f_config, "width", 1);
    print_json(f_config, "style", rgba_string(colors[i][0], colors[i][1], colors[i][2], 1));
    print_json(f_config, "anchor", "[0, 0]");
    print_json(f_config, "max-speed", 0);
    print_json(f_config, "speed-style", rgba_string(colors[i][0], colors[i][1], colors[i][2], 0.01));
    print_json(f_config, "vision-radius", 0);
    print_json(f_config, "vision-angle", 0);
    print_json(f_config, "vision-style", rgba_string(colors[i][0], colors[i][1], colors[i][2], 0.2));
    print_json(f_config, "attack-radius", 0);
    print_json(f_config, "attack-angle", 0);
    print_json(f_config, "attack-style", rgba_string(colors[i][0], colors[i][1], colors[i][2], 0.1));
    print_json(f_config, "broadcast-radius", 1, true);
    f_config << "}," << std::endl;

    // snake head
    i = 0;
    f_config << "{" << std::endl;
    print_json(f_config, "height", 1);
    print_json(f_config, "width", 1);
    print_json(f_config, "style", rgba_string(colors[i][0], colors[i][1], colors[i][2], 1));
    print_json(f_config, "anchor", "[0, 0]");
    print_json(f_config, "max-speed", 0);
    print_json(f_config, "speed-style", rgba_string(colors[i][0], colors[i][1], colors[i][2], 0.01));
    print_json(f_config, "vision-radius", 0);
    print_json(f_config, "vision-angle", 0);
    print_json(f_config, "vision-style", rgba_string(colors[i][0], colors[i][1], colors[i][2], 0.2));
    print_json(f_config, "attack-radius", 0);
    print_json(f_config, "attack-angle", 0);
    print_json(f_config, "attack-style", rgba_string(colors[i][0], colors[i][1], colors[i][2], 0.1));
    print_json(f_config, "broadcast-radius", 1, true);
    f_config << "}," << std::endl;

    // snake body
    i = 2;
    f_config << "{" << std::endl;
    print_json(f_config, "height", 1);
    print_json(f_config, "width", 1);
    print_json(f_config, "style", rgba_string(colors[i][0], colors[i][1], colors[i][2], 0.9));
    print_json(f_config, "anchor", "[0, 0]");
    print_json(f_config, "max-speed", 0);
    print_json(f_config, "speed-style", rgba_string(colors[i][0], colors[i][1], colors[i][2], 0.01));
    print_json(f_config, "vision-radius", 0);
    print_json(f_config, "vision-angle", 0);
    print_json(f_config, "vision-style", rgba_string(colors[i][0], colors[i][1], colors[i][2], 0.2));
    print_json(f_config, "attack-radius", 0);
    print_json(f_config, "attack-angle", 0);
    print_json(f_config, "attack-style", rgba_string(colors[i][0], colors[i][1], colors[i][2], 0.1));
    print_json(f_config, "broadcast-radius", 1, true);
    f_config << "}" << std::endl;

    f_config << "]" << std::endl;
    f_config << "}" << std::endl;

    /***** static *****/
    std::ofstream f_static(save_dir + "/" + "static.map");
    std::vector<Position> walls;
    map.get_wall(walls);

    // walls
    f_static << walls.size() << std::endl;
    for (int i = 0; i < walls.size(); i++) {
        f_static << walls[i].x << " " << walls[i].y << std::endl;
    }
}


void RenderGenerator::render_a_frame(const std::vector<Agent *> &agents, const std::set<Food *> &foods) {
    if (save_dir == "") {
        return;
    }

    std::string filename = save_dir + "/" + "video_" + std::to_string(file_ct) + ".txt";
    std::ofstream fout(filename.c_str(), frame_ct == 0 ? std::ios::out : std::ios::app);

    int num_snake = 0;
    for (int i = 0; i < agents.size(); i++) {
        if (agents[i]->is_dead())
            continue;
        num_snake += agents[i]->get_body().size();
    }
    
    int num_food = foods.size();
//    for (auto food : foods) {
//        int w, h;
//        food->get_size(w, h);
//        num_food += w * h;
//     }

    fout << "F" << " " << num_snake + num_food << " " << 0 << " " << 0 << std::endl;

    const int dir2angle[] = {0, 90, 180, 270};
    int hp = 100;
    int dir = dir2angle[3];

    // food
    for (auto food : foods) {
        int x, y;
        int width, height;
        food->get_xy(x, y);
        fout << id_ct++ << " " <<  hp << " " << dir << " " << x << " " << y << " " << 1 << std::endl;
//        food->get_size(width, height);
//        for (int i = 0; i < width; i++)
//            for (int j = 0; j < height; j++)
//                fout << id++ << " " <<  hp << " " << dir << " " << x+i << " " << y+j << " " << 1 << std::endl;
    }

    // agent
    for (auto agent : agents) {
        if (agent->is_dead())
            continue;
        auto &body = agent->get_body();
        int tmp = body.size();
        for (auto pos_iter = body.rbegin(); pos_iter != body.rend(); pos_iter++ ) {
            int color = --tmp == 0 ? 0 : 2;
            fout << id_ct++ << " " <<  hp << " " << dir << " " << pos_iter->x << " " << pos_iter->y
                 << " " << color << std::endl;
        }
    }

    if (frame_ct++ > frame_per_file) {
        frame_ct = 0;
        file_ct++;
    }
}

} // namespace discrete_snake
} // namespace magent
