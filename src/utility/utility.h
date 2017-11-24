/**
 * \file utility.h
 * \brief common utility for the project
 */

#ifndef MAGENT_UTILITY_H
#define MAGENT_UTILITY_H

#include <array>
#include <iostream>
#include <sstream>

namespace magent {
namespace utility {

// return true if the two strings are the same
bool strequ(const char *a, const char *b);

/**
 * a class for transforming linear pointer to multi dimensional pointer
 * e.g.  NDPointer<int, 3> multi(linear, {n, m, k}) means transforming "int *linear" to "int (*multi)[m][k]"
 *       the first dimension(n) is useless, so you can also use NDPointer<int, 3> multi(linear, {-1, m, k})
 */
template <typename T, int DIM>
class NDPointer {
public:
    NDPointer(T *data, const std::array<int, DIM> raw_dim) {
        this->data = data;

        int now = 1;
        for (int i = 0; i < DIM; i++) {
            multiplier[DIM - 1 - i] = now;
            now *= raw_dim[DIM - 1 - i];
        }
    }

    template <typename... TT>
    T &at(TT... arg_index) {
        int index = calc_index(arg_index...);
        return data[index];
    }
    T *data;

private:
    template <int n>
    int calc_index(int x) {
        return multiplier[n] * x;
    }

    template <int n=0, typename... TT>
    int calc_index(int x, TT... args) {
        return multiplier[n] * x + calc_index<n+1>(args...);
    }

    int multiplier[DIM];
};


/**
 * Simple Logger
 */
template <bool throw_exception=false>
class Logger
{
public:
    Logger(const char *filename, int line) {
        if (filename != nullptr)
            buffer << filename << ":" << line << " : ";
    }

    template <typename T>
    Logger& operator<<(T const & value) {
        buffer << value;
        return *this;
    }

    ~Logger() noexcept(false) {
        if (throw_exception)
            // dangerous, but the program is already fatal
            throw std::runtime_error(buffer.str());
        else {
            // This is atomic according to the POSIX standard
            // http://www.gnu.org/s/libc/manual/html_node/Streams-and-Threads.html
            std::cerr << buffer.str();
        }
    }

private:
    std::ostringstream buffer;
};

#define LOG(level) LOG_##level

#ifndef LOG_TRACE_ENABLE
#define LOG_TRACE_ENABLE 0
#endif

#ifndef LOG_WARNING_ENABLE
#define LOG_WARNING_ENABLE 0
#endif

#define LOG_TRACE   if (!LOG_TRACE_ENABLE) ; else magent::utility::Logger<>(nullptr, 0)
#define LOG_FATAL   magent::utility::Logger<true>(__FILE__, __LINE__)
#define LOG_ERROR   magent::utility::Logger<>(__FILE__, __LINE__)
#define LOG_WARNING if (!LOG_WARNING_ENABLE) ; else magent::utility::Logger<>(__FILE__, __LINE__)

} // namespace utility
} // namespace magent

#endif //MAGENT_UTILITY_H
