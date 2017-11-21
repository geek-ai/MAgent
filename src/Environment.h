/**
 * \file Environment.h
 * \brief Base class for environment
 */

#ifndef MAGNET_ENVIRONMENT_H
#define MAGNET_ENVIRONMENT_H

namespace magent {
namespace environment {

typedef int GroupHandle;

class Environment {
public:
    Environment() = default;
    virtual ~Environment() = default;

    // game
    virtual void set_config(const char *key, void *p_value) = 0;

    // run step
    virtual void reset() = 0;
    virtual void get_observation(GroupHandle group, float **linear_buffers) = 0;
    virtual void set_action(GroupHandle group, const int *actions) = 0;
    virtual void step(int *done) = 0;
    virtual void get_reward(GroupHandle group, float *buffer) = 0;

    // info getter
    virtual void get_info(GroupHandle group, const char *name, void *buffer) = 0;

    // render
    virtual void render() = 0;
};

typedef Environment* EnvHandle;

} // namespace environment
} // namespace magent

#endif //MAGNET_ENVIRONMENT_H
