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

const int RenderGenerator::cate_colors[20][3] = {
    {31, 119, 180},
    {174, 199, 232},
    {255, 127, 14},
    {255, 187, 120},
    {44, 160, 44},
    {152, 223, 138},
    {214, 39, 40},
    {255, 152, 150},
    {148, 103, 189},
    {197, 176, 213},
    {140, 86, 75},
    {196, 156, 148},
    {227, 119, 194},
    {247, 182, 210},
    {127, 127, 127},
    {199, 199, 199},
    {188, 189, 34},
    {219, 219, 141},
    {23, 190, 207},
    {158, 218, 229},
};

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
    for (int i = 0; i < sizeof(cate_colors)/sizeof(cate_colors[0]); i++) {
        f_config << "{" << std::endl;

        print_json(f_config, "height", 1);
        print_json(f_config, "width", 1);
        print_json(f_config, "style", rgba_string(cate_colors[i][0], cate_colors[i][1], cate_colors[i][2], 1));
        print_json(f_config, "anchor", "[0, 0]");
        print_json(f_config, "max-speed", (int)0);
        print_json(f_config, "speed-style", rgba_string(cate_colors[i][0], cate_colors[i][1], cate_colors[i][2], 0.01));
        print_json(f_config, "vision-radius", 0);
        print_json(f_config, "vision-angle", 0);
        print_json(f_config, "vision-style", rgba_string(cate_colors[i][0], cate_colors[i][1], cate_colors[i][2], 0.2));
        print_json(f_config, "attack-radius", 0);
        print_json(f_config, "attack-angle", 0);
        print_json(f_config, "attack-style", rgba_string(cate_colors[i][0], cate_colors[i][1], cate_colors[i][2], 0.1));
        print_json(f_config, "broadcast-radius", 1, true);

        if (i == sizeof(cate_colors)/sizeof(cate_colors[0]) - 1)
            f_config << "}" << std::endl;
        else
            f_config << "}," << std::endl;
    }
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

    std::string light_color[] = {
            " 255 0 0",
            " 0 255 0",
    };
    std::string building_colors[] = {
            " 205 205 205"
    };
    std::string light_tower_colors[] = {
            " 250 250 200"
    };
    std::string white_color = " 255 255 255";


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

    // count light tower
    int n_light_tower = 0;
    for (auto light : lights) {
        int status = light.get_status();
        int mask = light.get_mask();
        int x, y, w, h;
        std::tie(x, y, w, h) = light.get_location();
        if (mask & 0x1 || mask & 0x2)
            n_light_tower++;
        if (mask & 0x2 || mask & 0x4)
            n_light_tower++;
        if (mask & 0x4 || mask & 0x8)
            n_light_tower++;
        if (mask & 0x8 || mask & 0x1)
            n_light_tower++;
    }


    fout << "F" << " " << agents.size()
                << " " << 0                  // attack
                << " " << lights.size() * 2  // light lines
                << " " << buildings.size() + n_light_tower + parks.size() // rectangle
                << " " << 0                  // food
                << std::endl;

    const int dir2angle[] = {0, 90, 180, 270};
    int hp = -1;
    int dir = dir2angle[3];

    // draw point (car)
    for (auto agent : agents) {
        Position pos = agent->get_pos();
        fout << id_ct++ << " " << hp << " " << dir << " " << pos.x << " " << pos.y << " " << agent->get_color() << std::endl;
    }

    // draw lines
    for (auto light : lights) {
        int status = light.get_status();
        int x, y, w, h;
        int mask = light.get_mask();
        std::tie(x, y, w, h) = light.get_location();

        if (status == 0) {
            if (mask & 0x01)
                fout << "1" << " " << x+1  << " " << y+1 << " " << x + w << " " << y+1 << light_color[status] << std::endl;
            else
                fout << "1" << " " << x+1  << " " << y+1 << " " << x + w << " " << y+1 << white_color << std::endl;
            if (mask & 0x04)
                fout << "1" << " " << x+1 << " " << y + h << " " << x + w << " " << y + h << light_color[status] << std::endl;
            else
                fout << "1" << " " << x+1 << " " << y + h << " " << x + w << " " << y + h << white_color << std::endl;
        } else {
            if (mask & 0x02)
                fout << "1" << " " << x + w << " " << y+1 << " " << x + w << " " << y + h << light_color[1 - status] << std::endl;
            else
                fout << "1" << " " << x + w << " " << y+1 << " " << x + w << " " << y + h << white_color << std::endl;
            if (mask & 0x08)
                fout << "1" << " " << x+1 << " " << y+1 << " " << x+1 << " " << y + h << light_color[1 - status] << std::endl;
            else
                fout << "1" << " " << x+1 << " " << y+1 << " " << x+1 << " " << y + h << white_color << std::endl;
        }
    }

    if (frame_ct++ > frame_per_file) {
        frame_ct = 0;
        file_ct++;
    }

    // draw rectangles (building + light tower + park)
    for (auto building : buildings) {
        int x, y, w, h;
        std::tie(x, y, w, h) = building.get_location();
        fout << "2" << " " << x << " " << y << " " << x + w << " " << y + h << " " << building_colors[0] << std::endl;
    }

    // park
    for (int i = 0; i < parks.size(); i++) {
        int x, y, w, h;
        std::tie(x, y, w, h) = parks[i].get_location();
        fout << "2" << " " << x << " " << y << " " << x+w  << " " << y+h << " "
             << cate_colors[i][0] << " " << cate_colors[i][1] << " " << cate_colors[i][2] << std::endl;
    }

    // light tower
    std::string color;
    for (auto light : lights) {
        int status = light.get_status();
        int mask = light.get_mask();
        int x, y, w, h;
        std::tie(x, y, w, h) = light.get_location();
        if (mask & 0x1 || mask & 0x8)
            fout << "2" << " " << x << " " << y << " " << x + 1 << " " << y + 1 << " " << light_tower_colors[0] << std::endl;

        if (mask & 0x1 || mask & 0x2)
            fout << "2" << " " << x+w << " " << y << " " << x+w + 1 << " " << y + 1 << " " << light_tower_colors[0] << std::endl;

        if (mask & 0x8 || mask & 0x4)
            fout << "2" << " " << x << " " << y+h << " " << x + 1 << " " << y+h + 1 << " " << light_tower_colors[0] << std::endl;

        if (mask & 0x2 || mask & 0x4)
            fout << "2" << " " << x+w << " " << y+h << " " << x+w + 1 << " " << y+h + 1 << " " << light_tower_colors[0] << std::endl;
    }

}



} // namespace trans_city
} // namespace magent
