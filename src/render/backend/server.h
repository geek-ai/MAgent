#ifndef MAGNET_RENDER_BACKEND_SERVER_H_
#define MAGNET_RENDER_BACKEND_SERVER_H_

#include <fstream>

#include "utility/utility.h"
#include "protocol.h"
#include "socket.h"
#include "utility/config.h"
#include "utility/logger.h"

namespace magent {
namespace render {

template<class S, class P>
class TextServer : protected S {
private:
    static_assert(std::is_base_of<render::Base<std::string>, P>::value, "P is not derived from magent::render::Base<std::string>");
    static_assert(std::is_base_of<render::ISocket<uint16_t>, S>::value, "S is not derived from magent::render::ISocket");

    P protocol;
    render::Buffer buffer;
    render::Config config;

    void receive(const std::string & message) override {
        try {
            std::pair<const render::Type, const void * const> data = protocol.decode(message);
            switch (data.first) {
                case magent::render::LOAD:
                    load(
                            static_cast<const std::pair<std::string, std::string> * const>(data.second)->first,
                            static_cast<const std::pair<std::string, std::string> * const>(data.second)->second
                    );
                    delete(static_cast<const std::pair<std::string, std::string> * const>(data.second));
                    break;
                case magent::render::PICK:
                    const std::pair<const int, const render::Window> &coordinate =
                            *static_cast<const std::pair<const int, const render::Window> * const>(data.second);
                    pick(coordinate.first, coordinate.second);
                    delete(static_cast<const std::pair<const int, const render::Window> * const>(data.second));
                    break;
            }
        } catch (const render::RenderException &e) {
            render::Logger::STDERR.log(e.what());
            reply(e.what());
            return;
        }
    }

protected:
    void pick(int frame, const render::Window & window) {
        S::reply(protocol.encode(buffer[frame], config, buffer, window));
    }

    void load(const std::string & conf_path, const std::string & data_path) {
        std::ifstream handleConf(conf_path);
        try {
            config.load(handleConf);
            std::ifstream handleData(config.getDataPath() + '/' + data_path);
            buffer.load(handleData);
            reply(buffer.getFramesNumber());
        } catch (const magent::render::RenderException &e) {
            render::Logger::STDERR.log(e.what());
            reply(e.what());
        }
        handleConf.close();
    }

    /**
     * Reply frame data to the frontend.
     * @param frame: the frame to be send to the frontend.
     */
    void reply(const render::Frame & frame) {
        S::reply(protocol.encode(frame));
    }

    /**
     * Reply error message to the frontend.
     * @param message: the error message to be send to the frontend.
     */
    void reply(const std::string & message) override {
        S::reply(protocol.encodeError(message));
    }

    /**
     * Reply frame initial message to the frontend.
     * @param message: the initial data to be send to the frontend.
     */
    void reply(unsigned int nFrame) {
        S::reply(protocol.encode(config, nFrame));
    }

    void open() override {

    }

    void close() override {

    }

    void error() override {

    }

public:
    explicit TextServer(const RenderConfig &config, unsigned int maxBufferSize)
            : buffer(maxBufferSize), config(), S(config.port) {

    }

    void run() override {
        S::run();
    }
};

} // namespace render
} // namespace magent

#endif //MAGNET_RENDER_BACKEND_SERVER_H_
