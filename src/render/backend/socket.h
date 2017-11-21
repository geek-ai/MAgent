#ifndef MAGNET_RENDER_BACKEND_SOCKET_H_
#define MAGNET_RENDER_BACKEND_SOCKET_H_

#include <string>
#include <websocketpp/common/connection_hdl.hpp>

#include "utility/utility.h"

namespace magent {
namespace render {

template <class T>
class ISocket : public render::Unique {
protected:
    const T args;

public:
    explicit ISocket(const T &args) : args(args) {
    }

    virtual void reply(const std::string &) = 0;

    virtual void receive(const std::string &) = 0;

    virtual void open() = 0;

    virtual void close() = 0;

    virtual void error() = 0;

    virtual void run() = 0;
};

} // namespace render
} // namespace magent

#endif //MAGNET_RENDER_BACKEND_SOCKET_H_
