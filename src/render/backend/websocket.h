#ifndef MAGNET_RENDER_BACKEND_WEBSOCKET_H_
#define MAGNET_RENDER_BACKEND_WEBSOCKET_H_

#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/server.hpp>

#include "utility/logger.h"
#include "socket.h"

namespace magent {
namespace render {

class WebSocket : public ISocket<uint16_t> {
private:
    typedef websocketpp::server<websocketpp::config::asio_client> WSServer;

    WSServer ws;
    const websocketpp::connection_hdl * connection_hdl;

    void __on_open(const websocketpp::connection_hdl & /*connection*/);

    void __on_close(const websocketpp::connection_hdl & /*connection*/);

    void __on_error(const websocketpp::connection_hdl & /*connection*/);

    void __on_message(const websocketpp::connection_hdl & connection_hdl, WSServer::message_ptr message);

public:
    explicit WebSocket(uint16_t port);

    void reply(const std::string & message) override;

    void run() override;
};

} // namespace render
} // namespace magent

#endif //MAGNET_RENDER_BACKEND_WEBSOCKET_H_
