/**
 * \file RenderGenerator.cc
 * \brief Generate data for render
 */

#include "RenderGenerator.h"

#include <ios>
#include <fstream>
#include <string>
#include <tuple>

#include "RenderGenerator.h"
#include "TransCity.h"

namespace magent {
namespace trans_city {

RenderGenerator::RenderGenerator() {
    save_dir = "";
    file_ct = frame_ct = 0;
    frame_per_file = 1000000;

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

void RenderGenerator::gen_config(int w, int h) {
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

    // car
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

    // light
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

    // park
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
}


void RenderGenerator::render_a_frame(const std::vector<Agent *> &agents,
                                     const std::vector<Position> &walls,
                                     const std::vector<TrafficLight> &lights,
                                     const std::vector<Park> &parks,
                                     const std::vector<Building> &buildings) {
    if (save_dir == "")
        return;

    std::string filename = save_dir + "/" + "video_" + std::to_string(file_ct) + ".txt";
    std::ofstream fout(filename.c_str(), frame_ct == 0 ? std::ios::out : std::ios::app);

    // walls. only draw walls in the first frame
    if (frame_ct == 0) {
        // walls
        fout << "W" << " " << walls.size() << std::endl;
        for (int i = 0; i < walls.size(); i++) {
            fout << walls[i].x << " " << walls[i].y << std::endl;
        }
    }

    fout << "F" << " " << agents.size() + lights.size() * 4 + parks.size()
                << " " << 0                  // attack
                << " " << lights.size() * 4  // light lines
                << " " << 0                  // food
                << std::endl;

    const int dir2angle[] = {0, 90, 180, 270};
    int hp = 100;
    int dir = dir2angle[3];

    for (auto agent : agents) {
        Position pos = agent->get_pos();
        fout << id_ct++ << " " << hp << " " << dir << " " << pos.x << " " << pos.y << 0 << std::endl;
    }

    for (auto light : lights) {
        Position pos = light.get_pos();
        fout << id_ct++ << " " << hp << " " << dir << " " << pos.x << " " << pos.y << 0 << std::endl;
    }

    for (auto park : parks) {
        Position pos = park.get_pos();
        fout << id_ct++ << " " << hp << " " << dir << " " << pos.x << " " << pos.y << 0 << std::endl;
    }

    std::string color[] = {
            " 255 0 0",
            " 0 255 0",
    };
    for (auto light : lights) {
        int status = light.get_status();
        int x, y, w, h;
        std::tie(x, y, w, h) = light.get_location();
        //
        fout << "1" << " " << x << " " << y << " " << x + w << " " << y << color[status] << std::endl;
        fout << "1" << " " << x << " " << y + h << " " << x + w << " " << y + h << color[status] << std::endl;
        fout << "1" << " " << x << " " << y << " " << x << " " << y + h << color[1 - status] << std::endl;
        fout << "1" << " " << x + w << " " << y << " " << x + w << " " << y + h << color[1 - status] << std::endl;
    }

    if (frame_ct++ > frame_per_file) {
        frame_ct = 0;
        file_ct++;
    }
}



} // namespace trans_city
} // namespace magent
