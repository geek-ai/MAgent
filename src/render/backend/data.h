#ifndef MAGNET_RENDER_BACKEND_HANDLE_DATA_H_
#define MAGNET_RENDER_BACKEND_HANDLE_DATA_H_

#include <cstdio>
#include "utility/utility.h"
#include <vector>
#include <string>
#include <istream>

namespace magent {
namespace render {

struct Coordinate {
    int x, y;

    Coordinate();
    Coordinate(int x, int y);
};

struct AgentData : public render::Unique {
    Coordinate position;
    int id, hp, direction;
    unsigned int groupID;
};

struct BreadData : public render::Unique {
    Coordinate position;
    int hp;
};

struct EventData : public render::Unique {
    int type;
    const AgentData * agent;
    Coordinate position;
};

struct Window {
    Coordinate wmin, wmax;

    Window(int xmin, int ymin, int xmax, int ymax);

    bool accept(int, int)const;

    bool accept(int, int, int, int)const;
};

class Frame : public render::Unique {
private:
    unsigned int nAgents, nEvents, nBreads;
    render::AgentData * agents;
    render::EventData * events;
    render::BreadData * breads;

public:
    explicit Frame();

    void load(std::istream & /*handle*/);

    const unsigned int & getAgentsNumber() const;

    const unsigned int & getEventsNumber() const;

    const unsigned int & getBreadsNumber() const;

    const render::AgentData & getAgent(unsigned int id) const;

    const render::EventData & getEvent(unsigned int id) const;

    const render::BreadData & getBread(unsigned int id) const;

    ~Frame() override;

    void releaseMemory();
};

struct Style {
    unsigned int height, width, red, blue, green;
};

class Buffer : public render::Unique {
private:
    unsigned int nFrames, maxSize, nObstacles;
    Frame * frames;
    Coordinate * obstacles;

    void resize(unsigned int size);

public:
    explicit Buffer(unsigned int maxSize = 1000);

    void load(std::istream & /*handle*/);

    const Frame & operator [](unsigned int /*id*/)const;

    ~Buffer() override;

    const unsigned int & getFramesNumber()const;

    const unsigned int & getObstaclesNumber()const;

    const Coordinate & getObstacle(unsigned int id)const;

};

class Config : public render::Unique {
private:
    unsigned int height, width, miniMAPHeight, miniMAPWidth, nStyles;
    Style * styles;
    std::string frontendJSON;
    std::string dataPath;

public:
    explicit Config();

    void load(std::istream & /*handle*/);

    const unsigned int & getMiniMAPHeight()const;

    const unsigned int & getMiniMAPWidth()const;

    const unsigned int & getHeight()const;

    const unsigned int & getWidth()const;

    const unsigned int & getStylesNumber()const;

    const Style & getStyle(unsigned int id)const;

    ~Config() override;

    const std::string & getFrontendJSON()const;

    const std::string &getDataPath();
};

} // namespace render
} // namespace magent

#endif //MAGNET_RENDER_BACKEND_HANDLE_DATA_H_
