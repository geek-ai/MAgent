#include "logger.h"

namespace magent {
namespace render {

Logger Logger::STDOUT(std::cout);
Logger Logger::STDERR(std::cerr);
bool Logger::verbose = true;

void Logger::log(const std::string &rhs) {
    if (verbose) {
        std::time_t time = std::time(nullptr);
        std::unique_lock<std::mutex> locker(mutex);
        ostream << std::put_time(std::localtime(&time), "[%Y-%m-%d %H:%M:%S] [")
                << std::this_thread::get_id() << "] " << rhs << std::endl;
    }
}

void Logger::raw(const std::string &rhs) {
    if (verbose) {
        std::unique_lock<std::mutex> locker(mutex);
        ostream << rhs << std::endl;
    }
}

} // namespace magent
} // namespace render