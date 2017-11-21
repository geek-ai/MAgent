#include "websocket.h"

namespace magent {
namespace render {

void WebSocket::__on_open(const websocketpp::connection_hdl & /*unused*/) {
    ws.stop_listening();
    Logger::STDERR.log("successfully connected to the client");
    open();
}

void WebSocket::__on_close(const websocketpp::connection_hdl & /*unused*/) {
    Logger::STDERR.log("the client closed the connection");
    close();
#ifdef DEBUG
    ws.listen(args);
    ws.start_accept();
#else
    std::string answer;
    while (true) {
        std::cout << "Enter yes or no to continue or stop the server: ";
        std::cin >> answer;
        if (answer == "YES" || answer == "Yes" || answer == "yes") {
            ws.listen(args);
            ws.start_accept();
            break;
        }
        if (answer == "No" || answer == "no" || answer == "NO") {
            ws.stop();
            break;
        }
        std::cout << "Please enter yes or no to continue or stop the server.";
    }
#endif
}

void WebSocket::__on_error(const websocketpp::connection_hdl & /*unused*/) {
    error();
}

void WebSocket::__on_message(const websocketpp::connection_hdl &connection_hdl, WSServer::message_ptr message) {
    this->connection_hdl = &connection_hdl;
    Logger::STDERR.log("message receive: " + message->get_payload());
    receive(message->get_payload());
    this->connection_hdl = nullptr;
}

WebSocket::WebSocket(uint16_t port)  : ISocket(port), connection_hdl(nullptr) {
    ws.set_reuse_addr(true);
    ws.init_asio();
    ws.clear_access_channels(websocketpp::log::alevel::all);
    ws.clear_error_channels(websocketpp::log::alevel::all);
    ws.set_open_handler(websocketpp::lib::bind(&WebSocket::__on_open, this, websocketpp::lib::placeholders::_1));
    ws.set_close_handler(websocketpp::lib::bind(&WebSocket::__on_close, this, websocketpp::lib::placeholders::_1));
    ws.set_fail_handler(websocketpp::lib::bind(&WebSocket::__on_error, this, websocketpp::lib::placeholders::_1));
    ws.set_message_handler(websocketpp::lib::bind(
            &WebSocket::__on_message, this,
            websocketpp::lib::placeholders::_1,
            websocketpp::lib::placeholders::_2
    ));
}

void WebSocket::reply(const std::string &message) {
    websocketpp::lib::error_code errcode;
    ws.send(*connection_hdl, message, websocketpp::frame::opcode::text, errcode);
}

void WebSocket::run() {
    try {
        Logger::STDERR.log("Listening on port " + std::to_string(args));
        ws.listen(args);
        ws.start_accept();
        ws.run();
    } catch (const websocketpp::exception &e) {
        Logger::STDERR.log("cannot listen on port " + std::to_string(args) + " reason: " + e.what());
    }
}

} // namespace render
} // namespace magent