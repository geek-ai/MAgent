#ifndef MATRIX_RENDER_BACKEND_UTILITY_LOGGER_H_
#define MATRIX_RENDER_BACKEND_UTILITY_LOGGER_H_

#include <mutex>
#include <thread>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include "utility.h"

namespace magent {
namespace render {

/**
 * Logger is a thread-safe class to output message
 */
class Logger : public Unique {
private:
    std::mutex mutex;
    std::ostream & ostream;

    explicit Logger(std::ostream &ostream)  : ostream(ostream) {}

public:
    static Logger STDOUT;
    static Logger STDERR;
    static bool verbose;

    void log(const std::string &rhs);
    void raw(const std::string &rhs);
};

} // namespace render
} // namespace magent

#endif //MATRIX_RENDER_BACKEND_UTILITY_LOGGER_H_
