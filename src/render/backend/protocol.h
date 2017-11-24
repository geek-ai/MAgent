#ifndef MAGNET_RENDER_BACKEND_PROTOCOL_H_
#define MAGNET_RENDER_BACKEND_PROTOCOL_H_

#include <string>

#include "utility/exception.h"
#include "utility/utility.h"
#include "data.h"

namespace magent {
namespace render {

enum Type {
    LOAD,
    PICK
};

typedef const std::pair<const Type, const void * const> Result;

template<class T>
class Base : public render::Unique {
private:
    virtual T encode(const render::AgentData &)const = 0;

    virtual T encode(const render::EventData &)const = 0;

    virtual T encode(const render::BreadData &)const = 0;

    virtual T encode(const render::Config &, unsigned int)const = 0;

public:

    virtual T encode(const render::Frame &, const render::Config &,
                     const render::Buffer &, const render::Window &)const = 0;

    virtual T encodeError(const T &)const = 0;

    virtual const std::pair<const Type, const void * const> decode(const T &)const = 0;

};

} // namespace render
} // namespace magent

#endif //MAGNET_RENDER_BACKEND_PROTOCOL_H_