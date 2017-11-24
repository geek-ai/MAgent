/**
 * \file RenderGenerator.cc
 * \brief Generate data for render
 */

#include <ios>
#include <fstream>
#include <string>

#include "RenderGenerator.h"
#include "GridWorld.h"

namespace magent {
namespace gridworld {

RenderGenerator::RenderGenerator() {
    save_dir = "";
    file_ct = frame_ct = 0;
    frame_per_file = 10000;
}

void RenderGenerator::next_file() {
    file_ct++;
    frame_ct = 0;
}

void RenderGenerator::set_attack_event(std::vector<RenderAttackEvent> &attack_events) {
    this->attack_events = attack_events;
}

std::vector<RenderAttackEvent> &RenderGenerator::get_attack_event() {
    return this->attack_events;
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

void RenderGenerator::gen_config(std::vector<Group> &group, int w, int h) {
    /***** config *****/
    std::ofstream f_config(save_dir + "/" + "config.json");

    int colors[][3] = {
        {192, 64, 64},
        {64, 64, 192},
        {64, 192, 64},
        {64, 64, 64},
    };

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
    for (int i = 0; i < group.size(); i++) {
        AgentType &type = group[i].get_type();
        f_config << "{" << std::endl;

        print_json(f_config, "height", type.length);
        print_json(f_config, "width", type.width);
        print_json(f_config, "style", rgba_string(colors[i][0], colors[i][1], colors[i][2], 1));
        print_json(f_config, "anchor", "[0, 0]");
        print_json(f_config, "max-speed", (int)type.speed);
        print_json(f_config, "speed-style", rgba_string(colors[i][0], colors[i][1], colors[i][2], 0.01));
        print_json(f_config, "vision-radius", type.view_radius);
        print_json(f_config, "vision-angle", type.view_angle);
        print_json(f_config, "vision-style", rgba_string(colors[i][0], colors[i][1], colors[i][2], 0.2));
        print_json(f_config, "attack-radius", type.attack_radius);
        print_json(f_config, "attack-angle", type.attack_angle);
        print_json(f_config, "attack-style", rgba_string(colors[i][0], colors[i][1], colors[i][2], 0.1));
        print_json(f_config, "broadcast-radius", 1, true);

        if (i == group.size() - 1)
            f_config << "}" << std::endl;
        else
            f_config << "}," << std::endl;
    }
    f_config << "]" << std::endl;
    f_config << "}" << std::endl;
}


void RenderGenerator::render_a_frame(std::vector<Group> &groups, const Map &map) {
    if (save_dir == "") {
        return;
    }

    std::string filename = save_dir + "/" + "video_" + std::to_string(file_ct) + ".txt";
    std::ofstream fout(filename.c_str(), frame_ct == 0 ? std::ios::out : std::ios::app);

    // walls
    if (frame_ct ==0) {
        std::vector<Position> walls;
        map.get_wall(walls);

        // walls
        fout << "W" << " " << walls.size() << std::endl;
        for (int i = 0; i < walls.size(); i++) {
            fout << walls[i].x << " " << walls[i].y << std::endl;
        }
    }

    // count agents
    int num_agents = 0;
    for (int i  = 0; i < groups.size(); i++) {
        num_agents += groups[i].get_agents().size();
        bool can_absorb = groups[i].get_type().can_absorb;
        if (can_absorb) {
            const std::vector<Agent*> &agents = groups[i].get_agents();
            for (int j = 0; j < agents.size(); j++) {
                if (!agents[j]->is_absorbed()) {
                    num_agents--;
                }
            }
        }
    }
    int num_attacks = (int)attack_events.size();

    // frame info
    fout << "F" << " " << num_agents << " " << num_attacks << " " << 0 << std::endl;

    // agent
    const int dir2angle[] = {0, 90, 180, 270};
    for (int i  = 0; i < groups.size(); i++) {
        const std::vector<Agent*> &agents = groups[i].get_agents();

        bool can_absorb = agents[0]->get_type().can_absorb;
        // when type.can_absorb is true, do not render it if it is not absorbed

        for (int j = 0; j < agents.size(); j++) {
            const Agent &agent = *agents[j];

            if (can_absorb && !agent.is_absorbed())
                continue;

            Position pos = agent.get_pos();
            int id = agent.get_id();
            int hp = std::max(0, int(100 * agent.get_hp() / agent.get_type().hp));
            hp = std::min(hp, 100);
            int dir = dir2angle[(int)agent.get_dir()];

            fout << id << " " <<  hp << " " << dir << " " << pos.x << " " << pos.y << " " << i << std::endl;
        }
    }

    // attack event
    for (int i = 0; i < attack_events.size(); i++) {
        int op = 0;
        int id = attack_events[i].id;
        int x  = attack_events[i].x;
        int y  = attack_events[i].y;

        fout << op << " " << id << " " << x << " " << y << std::endl;
    }

    if (frame_ct++ > frame_per_file) {
        frame_ct = 0;
        file_ct++;
    }
}

} // namespace gridworld
} // namespace magent
