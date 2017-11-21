#ifndef MAGNET_RENDER_BACKEND_UTILITY_UNIQUE_H_
#define MAGNET_RENDER_BACKEND_UTILITY_UNIQUE_H_

namespace magent {
namespace render {

class Unique {
public:
    Unique() = default;

    Unique(const Unique &) = delete;

    Unique(const Unique &&) = delete;

    Unique &operator =(const Unique &) = delete;

    Unique &operator =(const Unique &&) = delete;

    virtual ~Unique() = default;
};

} // namespace render
} // namespace magent

#endif //MAGNET_RENDER_BACKEND_UTILITY_UNIQUE_H_
