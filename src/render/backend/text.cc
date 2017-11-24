#ifndef MAGNET_RENDER_BACKEND_TEXT_CPP_
#define MAGNET_RENDER_BACKEND_TEXT_CPP_

#include <cmath>
#include <cstring>
#include <unordered_map>
#include "text.h"
#include "utility/logger.h"

namespace magent {
namespace render {

std::string Text::encode(const magent::render::AgentData &agent)const {
    return std::to_string(agent.id)
           + ' ' + std::to_string(agent.position.x)
           + ' ' + std::to_string(agent.position.y)
           + ' ' + std::to_string(agent.groupID)
           + ' ' + std::to_string(agent.direction)
           + ' ' + std::to_string(agent.hp);
}

std::string Text::encode(const magent::render::EventData &event)const {
    return std::to_string(event.type)
           + ' ' + std::to_string(event.agent->id)
           + ' ' + std::to_string(event.position.x)
           + ' ' + std::to_string(event.position.y);
}

std::string Text::encode(const render::Config &config, unsigned int nFrame)const {
    return 'i' + std::to_string(nFrame) + '|' + config.getFrontendJSON();
}

std::string Text::encodeError(const std::string &message)const {
    return 'e' + message;
}

Result Text::decode(const std::string &data)const {
    switch (data[0]) {
        case 'l': {
            long pos = data.find_first_of(',');
            if (pos == std::string::npos) {
                throw RenderException("invalid load operation");
            }
            return {
                    Type::LOAD,
                    new std::pair<std::string, std::string>(
                            data.substr(1, static_cast<unsigned long>(pos - 1)),
                            data.substr(static_cast<unsigned long>(pos + 1), data.length() - pos)
                    )
            };
        }
        case 'p': {
            int frameID, xmin, xmax, ymin, ymax;
            if (sscanf(data.substr(1).c_str(), "%d%d%d%d%d", &frameID, &xmin, &ymin, &xmax, &ymax) != 5) {
                throw RenderException("invalid pick operation");
            }
            return {
                    Type::PICK,
                    new std::pair<const int, const render::Window>(frameID, render::Window(xmin, ymin, xmax, ymax))
            };
        }
        default:
            throw RenderException("invalid message");
    }
}

std::string Text::encode(const magent::render::Frame &frame,
                         const magent::render::Config &config,
                         const magent::render::Buffer &buffer,
                         const magent::render::Window &window)const {
    std::string result("f");
    std::unordered_map<int, bool> hasEvent;
    for (unsigned int i = 0, size = frame.getEventsNumber(), first = 1; i < size; i++) {
        const render::EventData &data = frame.getEvent(i);
        const render::Style &style = config.getStyle(data.agent->groupID);
        unsigned int width = style.width;
        unsigned int height = style.height;
        if (data.agent->direction % 180 != 0) std::swap(width, height);
        if (window.accept(data.position.x, data.position.y)
            || window.accept(data.agent->position.x, data.agent->position.y, width, height)) {
            hasEvent[data.agent->id] = true;
            if (first == 0u) result.append("|");
            result.append(encode(data));
            first = 0;
        }
    }
    result.append(";");

    unsigned int mapHeight = config.getHeight();
    unsigned int mapWidth = config.getWidth();
    unsigned int miniMAPHeight = config.getMiniMAPHeight();
    unsigned int miniMAPWidth = config.getMiniMAPWidth();
    const unsigned int & nStyles = config.getStylesNumber();
    auto minimap = new unsigned int*[miniMAPHeight * miniMAPWidth];
    auto agentsCounter = new unsigned int[nStyles];
    for (unsigned int i = 0; i < miniMAPHeight * miniMAPWidth; i++) {
        minimap[i] = new unsigned int[nStyles];
        std::fill(minimap[i], minimap[i] + nStyles, 0);
    }
    memset(agentsCounter, 0, sizeof(*agentsCounter) * nStyles);
    for (unsigned int i = 0, size = frame.getAgentsNumber(), first = 1; i < size; i++) {
        const magent::render::AgentData & data = frame.getAgent(i);
        const render::Style &style = config.getStyle(data.groupID);
        unsigned int width = style.width;
        unsigned int height = style.height;
        if (data.direction % 180 != 0) std::swap(width, height);
        if (hasEvent[data.id] || window.accept(data.position.x, data.position.y, width, height)) {
            if (first == 0) result.append("|");
            result.append(encode(data));
            first = 0;
        }
        agentsCounter[data.groupID]++;

        auto miniPositionX = static_cast<unsigned int>(1.0 * data.position.x / mapWidth * miniMAPWidth);
        auto miniPositionY = static_cast<unsigned int>(1.0 * data.position.y / mapHeight * miniMAPHeight);
        minimap[miniPositionY * miniMAPWidth + miniPositionX][data.groupID]++;
    }
    result.append(";");

    for (unsigned int i = 0, size = frame.getBreadsNumber(), first = 1; i < size; i++) {
        const magent::render::BreadData & data = frame.getBread(i);
        if (window.accept(data.position.x, data.position.y)) {
            if (first == 0u) result.append("|");
            result.append(encode(data));
            first = 0;
        }
    }
    result.append(";");

    for (unsigned int i = 0, size = buffer.getObstaclesNumber(), first = 1; i < size; i++) {
        const render::Coordinate &now = buffer.getObstacle(i);
        if (window.accept(now.x, now.y)) {
            if (first == 0u) result.append("|");
            result.append(std::to_string(now.x));
            result.append(" ");
            result.append(std::to_string(now.y));
            first = 0;
        }
    }
    result.append(";");

    for (unsigned int i = 0, first = 1; i < miniMAPHeight * miniMAPWidth; i++) {
        if (first == 0u) result.append(" ");
        double red = 0, blue = 0, green = 0;
        unsigned int sum = 0;
        for (unsigned int j = 0; j < nStyles; j++) {
            sum += minimap[i][j];
        }
        for (unsigned int j = 0; j < nStyles; j++) {
            red += 1.0 * config.getStyle(j).red * minimap[i][j] / sum;
            blue += 1.0 * config.getStyle(j).blue * minimap[i][j] / sum;
            green += 1.0 * config.getStyle(j).green * minimap[i][j] / sum;
        }
        unsigned int value = 0;
        if (sum == 0u) {
            value = (0xFFu << 24) | (0xFFu << 16) | (0xFFu << 8) | (0xFFu << 0);
        } else {
            value |= static_cast<unsigned int>(red) << 24;
            value |= static_cast<unsigned int>(blue) << 16;
            value |= static_cast<unsigned int>(green) << 8;
            value |= static_cast<unsigned int>(0xFFu) << 0;
        }

        result.append(std::to_string(value));
        first = 0;
        delete[](minimap[i]);
    }
    delete[](minimap);

    result.append(";");
    for (unsigned int i = 0, first = 1; i < nStyles; i++) {
        if (first == 0u) result.append(" ");
        result.append(std::to_string(agentsCounter[i]));
        first = 0;
    }

    return result;
}

std::string Text::encode(const magent::render::BreadData &bread) const {
    return std::to_string(bread.position.x)
           + ' ' + std::to_string(bread.position.y)
           + ' ' + std::to_string(bread.hp);
}

} // namespace render
} // namespace magent

#endif // MAGNET_RENDER_BACKEND_TEXT_CPP_