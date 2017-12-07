cmake_minimum_required(VERSION 3.0)

project(magent)

IF (APPLE)
    set(CMAKE_CXX_COMPILER "/usr/local/opt/llvm/bin/clang++")
    link_directories("/usr/local/opt/llvm/lib")
    include_directories( "/usr/local/include" )
    link_directories("/usr/local/lib/")
ENDIF()

file(GLOB autopilot_sources src/*.cc src/gridworld/*.cc src/discrete_snake/*.cc src/utility/*.cc)
set(LIB_SRC_FILES ${autopilot_sources})

file(GLOB autopilot_sources src/render/*.cc src/render/*/*.cc src/render/*/*/*.cc)
set(RENDER_SRC_FILES ${autopilot_sources})

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -std=c++11 -O3")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-reorder -Wno-sign-compare -Wno-missing-braces")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread -DDEBUG")

# runtime library
add_library(magent SHARED ${LIB_SRC_FILES})
add_executable(testlib ${LIB_SRC_FILES})

# render
add_executable(render ${RENDER_SRC_FILES})

IF (APPLE)
    target_link_libraries(render boost_system jsoncpp argp omp)
ELSE()
    target_link_libraries(render boost_system jsoncpp)
ENDIF()

set_target_properties( render
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/render/")

# copy frontend and demo
add_custom_command(
        TARGET render POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/src/render/frontend/" "${CMAKE_BINARY_DIR}/render"
)

add_custom_command(
        TARGET render POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/src/render/backend/demo/" "${CMAKE_BINARY_DIR}/render/"
)
