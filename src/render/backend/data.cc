#include <cstring>
#include <fstream>
#include <sstream>
#ifdef __linux__
    #include <jsoncpp/json/json.h>
#else
    #include <json/json.h>
#endif
#include <unordered_map>

#include "data.h"
#include "utility/logger.h"
#include "utility/exception.h"

namespace magent {
namespace render {


Window::Window(int xmin, int ymin, int xmax, int ymax) : wmin(xmin, ymin), wmax(xmax, ymax) {

}

bool Window::accept(int x, int y) const {
    return wmin.x <= x && wmin.y <= y && x <= wmax.x && y <= wmax.y;
}

bool Window::accept(int x, int y, int w, int h) const {
    return ((wmin.x <= x && x <= wmax.x) || (wmin.x <= x + w && x + w <= wmax.x))
           && ((wmin.y <= y && y <= wmax.y) || (wmin.y <= y + h && y + h <= wmax.y));
}

Coordinate::Coordinate(int x, int y) : x(x), y(y) {

}

Coordinate::Coordinate() : x(0), y(0) {

}

const unsigned int &Frame::getAgentsNumber() const {
    return nAgents;
}

const unsigned int &Frame::getEventsNumber() const {
    return nEvents;
}

const AgentData &Frame::getAgent(unsigned int id) const {
    return agents[id];
}

const EventData &Frame::getEvent(unsigned int id) const {
    return events[id];
}

void Frame::load(std::istream & handle) {
    if (!handle) {
        throw RenderException("invalid file handle");
    }
    if (!(handle >> nAgents >> nEvents >> nBreads)) {
        throw RenderException("cannot read nAgents and nEvents and nBreads");
    }
    if (agents != nullptr) {
        delete[](agents);
        agents = nullptr;
    }

    if (events != nullptr) {
        delete[](events);
        events = nullptr;
    }

    if (breads != nullptr) {
        delete[](breads);
        breads = nullptr;
    }

    agents = new render::AgentData[nAgents];
    std::unordered_map<int, int> map;
    try {
        for (unsigned int i = 0; i < nAgents; i++) {
            if (!(handle >> agents[i].id >> agents[i].hp >> agents[i].direction
                         >> agents[i].position.x >> agents[i].position.y >> agents[i].groupID)) {
                throw RenderException("cannot read the next agent, map file may be broken");
            }
            map[agents[i].id] = i;
        }
    } catch (const RenderException &e) {
        delete[](agents);
        agents = nullptr;
        nAgents = 0;
        throw;
    }

    events = new render::EventData[nEvents];
    try{
        for (unsigned int i = 0, j = 0; i < nEvents; i++) {
            int id;
            if (!(handle >> events[i].type >> id >> events[i].position.x >> events[i].position.y)) {
                throw RenderException("cannot read the next event, map file may be broken");
            }
            if (map.find(id) != map.end()) {
                events[i].agent = &agents[map[id]];
            }
        }
    } catch (const RenderException &e) {
        delete[](events);
        events = nullptr;
        nEvents = 0;
        throw;
    }

    breads = new render::BreadData[nBreads];
    try{
        for (unsigned int i = 0; i < nBreads; i++) {
            if (!(handle >> breads[i].position.x >> breads[i].position.y >> breads[i].hp)) {
                throw RenderException("cannot read the next food, map file may be broken");
            }
        }
    } catch (const RenderException &e) {
        delete[](breads);
        breads = nullptr;
        nBreads = 0;
        throw;
    }
}

Frame::~Frame() {
    if (agents != nullptr) {
        delete[](agents);
        agents = nullptr;
    }

    if (events != nullptr) {
        delete[](events);
        events = nullptr;
    }

    if (breads != nullptr) {
        delete[](breads);
        breads = nullptr;
    }
}

Frame::Frame() : nEvents(0), nAgents(0), nBreads(0), agents(nullptr), events(nullptr), breads(nullptr) {

}

const unsigned int &Frame::getBreadsNumber() const {
    return nBreads;
}

const BreadData &Frame::getBread(unsigned int id) const {
    return breads[id];
}

void Frame::releaseMemory() {
    agents = nullptr;
    events = nullptr;
    breads = nullptr;
}

Buffer::~Buffer() {
    maxSize = 0;
    if (frames != nullptr) {
        delete[](frames);
        frames = nullptr;
    }
    if (obstacles != nullptr) {
        delete[](obstacles);
        obstacles = nullptr;
    }
}

const Frame &Buffer::operator [](unsigned int id)const {
    return frames[id];
}

void Buffer::resize(unsigned int size) {
    auto memory = new Frame[size];
    memcpy(static_cast<void *>(memory), static_cast<void *>(frames), sizeof(*frames) * std::min(maxSize, size));
    for (unsigned int i = 0; i < maxSize; i++) {
        frames[i].releaseMemory();
    }
    delete[](frames);
    frames = memory;
    maxSize = size;
}

Buffer::Buffer(unsigned int maxSize) : maxSize(maxSize), frames(new Frame [maxSize]), nObstacles(0), obstacles(nullptr) {

}

const unsigned int & Buffer::getFramesNumber()const {
    return nFrames;
}

void Buffer::load(std::istream &handle) {
    if (!handle) {
        throw RenderException("invalid handle of the map data file");
    }

    std::string tmp;
    if (!(handle >> tmp >> nObstacles)) {
        throw RenderException("cannot read the number of obstacles in the data file");
    }

    if (tmp != "W") {
        throw RenderException("invalid tag of walls");
    }

    if (obstacles != nullptr) {
        delete[](obstacles);
        obstacles = nullptr;
    }
    try {
        obstacles = new Coordinate[nObstacles];
        for (unsigned int i = 0; i < nObstacles; i++) {
            if (!(handle >> obstacles[i].x >> obstacles[i].y)) {
                throw RenderException("cannot read the information of the next obstacle in the data file");
            }
        }
    } catch (const RenderException &e) {
        delete[](obstacles);
        obstacles = nullptr;
        throw;
    }

    nFrames = 0;
    while (!handle.eof()) {
        if (nFrames == maxSize) {
            resize(maxSize * 2);
        }
        if (!(handle >> tmp)) {
            break;
        }
        if (tmp != "F") {
            throw RenderException("invalid frame flag, the map file may be broken");
        }
        frames[nFrames++].load(handle);
    }
}

const unsigned int &Buffer::getObstaclesNumber() const {
    return nObstacles;
}

const Coordinate &Buffer::getObstacle(unsigned int id) const {
    return obstacles[id];
}

void Config::load(std::istream &handle) {
    if (!handle) {
        throw RenderException("invalid handle of the map configuration file");
    }
    Json::Reader reader;
    Json::Value root;
    if (reader.parse(handle, root)) {
        if (root["width"].isUInt()) {
            width = root["width"].asUInt();
        } else {
            throw RenderException("property width must be an UInt");
        }
        if (root["height"].isUInt()) {
            height = root["height"].asUInt();
        } else {
            throw RenderException("property height must be an UInt");
        }
        // if (root["static-file"].isString()) {
        // } else {
        //     throw RenderException("property height must be a String");
        // }
        if (root["minimap-width"].isUInt()) {
            miniMAPWidth = root["minimap-width"].asUInt();
        } else {
            throw RenderException("property minimap-width must be an UInt");
        }
        if (root["minimap-height"].isUInt()) {
            miniMAPHeight = root["minimap-height"].asUInt();
        } else {
            throw RenderException("property minimap-height must be an UInt");
        }
        if (!root["obstacle-style"].isString()) {
            throw RenderException("property obstacle-style must be an String");
        }
        if (!root["dynamic-file-directory"].isString()) {
            throw RenderException("property dynamic-file-directory must be an String");
        } else {
            dataPath = root["dynamic-file-directory"].asString();
            root.removeMember("dynamic-file-directory");
        }
        if (!root["attack-style"].isString()) {
            throw RenderException("property attack-style must be an UInt");
        }
        Json::Value & groups = root["group"];
        if (!groups) {
            throw RenderException("lacks of property groups");
        }
        if (groups.empty()) {
            throw RenderException("property groups must contain at least one group");
        }
        nStyles = groups.size();
        styles = new Style[nStyles];
        try {
            unsigned int index = 0;
            for (auto group : groups) {
                if (group["height"].isUInt()) {
                    styles[index].height = group["height"].asUInt();
                } else {
                    throw RenderException("property height in the group must be an UInt");
                }
                if (group["width"].isUInt()) {
                    styles[index].width = group["width"].asUInt();
                } else {
                    throw RenderException("property width in the group must be an UInt");
                }
                if (group["style"].isString()) {
                    if (sscanf(
                            group["style"].asString().c_str(),
                            "rgba(%d,%d,%d,%*f)",
                            &styles[index].red,
                            &styles[index].blue,
                            &styles[index].green
                    ) != 3) {
                        throw RenderException("property style in the group must be rgba(r: int, g: int, b: int, a: float)");
                    }
                } else {
                    throw RenderException("property style in the group must be an String");
                }
                if (!group["anchor"].isArray() || group["anchor"].size() != 2u) {
                    throw RenderException("property turn must be an array of size 2");
                } else {
                    // FIXME: maybe load the group information here
                }
                if (group["max-speed"].isUInt()) {
                    // FIXME: maybe load the group information here
                } else {
                    throw RenderException("property max-speed must be an UInt");
                }
                if (group["vision-radius"].isNumeric()) {
                    // FIXME: maybe load the group information here
                } else {
                    throw RenderException("property vision-radius must be a number");
                }
                if (group["vision-angle"].isUInt()) {
                    // FIXME: maybe load the group information here
                } else {
                    throw RenderException("property vision-angle must be an UInt");
                }
                if (group["vision-style"].isString()) {
                    // FIXME: maybe load the group information here
                } else {
                    throw RenderException("property vision-style must be an UInt");
                }
                if (group["attack-radius"].isNumeric()) {
                    // FIXME: maybe load the group information here
                } else {
                    throw RenderException("property attack-radius must be a number");
                }
                if (group["attack-angle"].isUInt()) {
                    // FIXME: maybe load the group information here
                } else {
                    throw RenderException("property attack-angle must be an UInt");
                }
                if (group["broadcast-radius"].isNumeric()) {
                    // FIXME: maybe load the group information here
                } else {
                    throw RenderException("property broadcast-radius must be a number");
                }
                index++;
            }
        } catch (const RenderException &e) {
            delete[](styles);
            styles = nullptr;
            throw;
        }
        Json::FastWriter writer;
        frontendJSON = writer.write(root);
    } else {
        throw RenderException("validation to JSON configuration file failed");
    }
}

Config::~Config() {
    if (styles != nullptr){
        delete[](styles);
        styles = nullptr;
    }
}

const std::string & Config::getFrontendJSON()const {
    return frontendJSON;
}

const std::string &Config::getDataPath() {
    return dataPath;
}

const Style &Config::getStyle(unsigned int id) const {
    return styles[id];
}

const unsigned int &Config::getStylesNumber() const {
    return nStyles;
}

Config::Config()
        : height(0), width(0), nStyles(0), styles(nullptr){

}

const unsigned int &Config::getHeight() const {
    return height;
}

const unsigned int &Config::getWidth() const {
    return width;
}

const unsigned int &Config::getMiniMAPHeight() const {
    return miniMAPHeight;
}

const unsigned int &Config::getMiniMAPWidth() const {
    return miniMAPWidth;
}


} // namespace render
} // namespace magent