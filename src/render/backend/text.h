#ifndef MAGNET_RENDER_BACKEND_PROTOCOL_TEXT_H_
#define MAGNET_RENDER_BACKEND_PROTOCOL_TEXT_H_

#include <string>

#include "protocol.h"
#include "data.h"

namespace magent {
namespace render {

class Text : public Base<std::string> {
private:
    std::string encode(const render::AgentData & /*unused*/)const override;

    std::string encode(const render::EventData & /*unused*/)const override;

    std::string encode(const render::BreadData & /*unused*/)const override ;

public:
    std::string encode(const render::Config & /*unused*/, unsigned int /*unused*/)const override;

    std::string encode(const render::Frame & /*unused*/,
                       const render::Config & /*unused*/,
                       const render::Buffer & /*unused*/,
                       const render::Window & /*unused*/)const override;

    std::string encodeError(const std::string & /*unused*/)const override;

    Result decode(const std::string & /*unused*/)const override;
};

} // namespace render
} // namespace magent

#endif //MAGNET_RENDER_BACKEND_PROTOCOL_TEXT_H_
