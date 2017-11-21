/**
 * \file DiscreteSnake.h
 * \brief core game engine of the discrete snake
 */

#ifndef MAGNET_DISCRETE_SNACK_H
#define MAGNET_DISCRETE_SNACK_H

#include <cstring>
#include <deque>
#include <set>
#include "snake_def.h"
#include "Map.h"
#include "RenderGenerator.h"

namespace magent {
namespace discrete_snake {

class DiscreteSnake : public Environment {
public:
    DiscreteSnake();
    ~DiscreteSnake() override;

    // game
    void set_config(const char *key, void * p_value) override;

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
    Map map;
    std::vector<Agent*> agents;
    std::set<Food*> foods;
    int *head_mask;

    int id_counter;
    bool first_render;

    /* config */
    int width, height;
    int view_width, view_height;
    Reward max_dead_penalty, corpse_value;
    int embedding_size;
    int initial_length;
    int total_resource;

    /* render */
    RenderGenerator render_generator;
};

class Food {
public:
    Food(int w, int h, float v) : width(w), height(h), value(v) {
    }

    void set_xy(int x, int y) {
        this->x = x;
        this->y = y;
    }
    void get_xy(int &x, int &y) const {
        x = this->x;
        y = this->y;
    }

    void get_size(int &w, int &h) const {
        w = width;
        h = height;
    }
    float get_value() const { return value; }


private:
    int x, y;
    int width, height;
    float value;
};

class Agent {
public:
    explicit Agent(int &id_counter): group(0), dead(false), in_event_calc(false), dir(DIR_NUM), last_action(ACT_NUM),
                                     next_reward(0), total_reward(0) {
        id = group;
        id = id_counter++;
    }

    void init_body(std::vector<Position> &pos) {
        body.clear();
        body.insert(body.begin(), pos.begin(), pos.end());
        total_reward = 0;
    }

    void set_dir(Direction dir) { this->dir = dir; }
    Direction get_dir() const { return dir; }

    void set_action(Action act) { this->last_action = act; }
    Action get_action()  const   { return this->last_action; }

    void init_reward() { next_reward = 0; }
    void add_reward(Reward r)  {
        next_reward += r;
        total_reward += r;
    }
    Reward get_reward() const { return next_reward; }
    Reward get_total_reward() const { return total_reward; }

    Position get_head() const { return body.front(); }
    void push_head(Position head) { body.push_front(head); }
    Position pop_tail() {
        Position ret =  body.back();
        body.pop_back();
        return ret;
    }

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

    std::deque<Position> &get_body() { return body; }
    size_t get_length() const { return body.size(); }

    void set_dead() { dead = true; }
    bool is_dead() { return dead; }

private:
    std::deque<Position> body;
    Direction dir;

    Reward next_reward;
    Reward total_reward;
    Action last_action;
    bool dead;
    bool in_event_calc;

    int id;
    std::vector<float> embedding;
    GroupHandle group;
};

} // namespace discrete_snake
} // namespace magent

#endif //MAGNET_DISCRETE_SNACK_H
