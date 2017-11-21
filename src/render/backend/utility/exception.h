#ifndef MAGNET_RENDER_BACKEND_UTILITY_EXCEPTION_H_
#define MAGNET_RENDER_BACKEND_UTILITY_EXCEPTION_H_

#include <string>
#include <stdexcept>
#include "utility.h"

namespace magent {
namespace render {

class BaseException : public std::runtime_error {
public:
    explicit BaseException(const std::string &message) : std::runtime_error(message) {}

    explicit BaseException(const char *message) : std::runtime_error(message) {}
};

class RenderException : public BaseException {
public:
    explicit RenderException(const std::string &message) :
        BaseException("magent::render::RenderException: \"" + message + "\"") {}

    explicit RenderException(const char *message) :
        BaseException("magent::render::RenderException: \"" + std::string(message) + "\"") {}
};

} // namespace render
} // namespace magent

#endif //MAGNET_RENDER_BACKEND_UTILITY_EXCEPTION_H_
