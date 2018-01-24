/**
 * \file TransCity.h
 * \brief core game engine of the trans city
 */

#ifndef MAGENT_TRANSCITY_TRANSCITY_H
#define MAGENT_TRANSCITY_TRANSCITY_H

#include <cstring>
#include <deque>
#include <set>
#include <vector>
#include <tuple>
#include <random>

#include "city_def.h"
#include "Map.h"
#include "RenderGenerator.h"

namespace magent {
namespace trans_city {

class TransCity : public Environment {
public:
    TransCity();
    ~TransCity() override;

    // game
    void set_config(const char *key, void *p_value) override;

    // run step
    void reset() override;
    void get_observation(GroupHandle group, float **buffer) override;
    void set_action(GroupHandle group, const int *actions) override;
    void step(int *done) override;
    void get_reward(GroupHandle group, float *buffer) override;

    // info getter
    void get_info(GroupHandle group, const char *name, void *void_buffer) override;

    // render
    void render() override;
    void render_next_file();

    void clear_dead();

    void add_object(int obj_id, int n, const char *method, const int *linear_buffer);

private:
    int get_feature_size_(GroupHandle group);

    // game config
    int width, height;
    int view_width, view_height;
    int embedding_size;
    int interval_max, interval_min;

    // game status: map, agent,
    Map map;
    std::vector<Agent*> agents;
    std::vector<TrafficLight> lights;
    std::vector<Park> parks;
    std::vector<Position> walls;
    std::vector<Building> buildings;
    std::default_random_engine random_engine;

    int id_counter{0};

    // render
    bool first_render{true};
    RenderGenerator render_generator;
};


class Park {
public:
    Park(Position p, int w, int h) : pos(p), width(w), height(h) {}

    Position get_pos() { return pos; }
private:
    Position pos;
    int width, height;
};


class TrafficLight {
public:
    TrafficLight(Position p, int w, int h, int v):
            pos(p), width(w), height(h), interval(v) {
        counter = static_cast<int>(random()) % (2 * interval);
        update_status();
    }

    void update_status() {
        counter = (counter + 1) % (2 * interval);
        status = counter / (2 * interval);
    }

    int get_status() {
        return status;
    }

    Position get_pos() { return pos; }
    std::tuple<int, int, int, int> get_location() { return std::make_tuple(pos.x, pos.y, width, height); };

private:
    Position pos;
    int width, height, interval;
    int status;  // 0 - red, 1 - green
    int counter;
};

class Building {
public:
    Building(Position p, int w, int h):
            pos(p), width(w), height(h) {
    }

    std::tuple<int, int, int, int> get_location() { return std::make_tuple(pos.x, pos.y, width, height); }

private:
    Position pos;
    int width, height, interval;
};

class Agent {
public:
    explicit Agent(int &id_counter): group(0), dead(false),
                                     last_action(ACT_NUM), next_reward(0) {
        id = id_counter++;

        last_action = static_cast<Action>(ACT_NUM); // dangerous here !
        next_reward = 0;

        init_reward();
    }

    void set_action(Action act) { this->last_action = act; }
    Action get_action()  const  { return this->last_action; }

    void init_reward() {
        last_reward = next_reward;
        next_reward = 0;
    }
    Reward get_reward()         { return next_reward; }
    Reward get_last_reward()    { return last_reward; }
    void add_reward(Reward add) { next_reward += add; }

    int get_id() const { return id; }
    void get_embedding(float *buf, int size) {
        if (embedding.empty()) {
            int t = id;
            for (int i = 0; i < size; i++, t >>= 1) {
                embedding.push_back((float)(t & 1));
            }
        }
        memcpy(buf, &embedding[0], sizeof(float) * size);
    }

    Position &get_pos()             { return pos; }
    const Position &get_pos() const { return pos; }
    void set_pos(Position pos) { this->pos = pos; }

    Position get_goal() const { return goal; }
    void set_goal(Position center) { goal = center; }

    void set_dead() { dead = true; }
    bool is_dead()  { return dead; }

private:
    int id;
    bool dead;

    Position goal, pos;

    Action last_action;
    Reward next_reward, last_reward;

    std::vector<float> embedding;
    GroupHandle group;
};

} // namespace trans_city
} // namespace magent


#endif //MAGENT_TRANSCITY_TRANSCITY_H
