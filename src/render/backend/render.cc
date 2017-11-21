#include "server.h"
#include "websocket.h"
#include "text.h"

int main(int argc, char *argv[]) {
    magent::render::RenderConfig config;
    magent::render::parse(argc, argv, config);
    magent::render::Logger::verbose = !config.quiet;
    magent::render::TextServer<magent::render::WebSocket, magent::render::Text> server(config, 256);

    server.run();
}
