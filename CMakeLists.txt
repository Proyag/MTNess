cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(mtness)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Git QUIET)
execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive --no-fetch
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(src
                    extern
                    extern/spdlog/include
                    extern/sentencepiece
                    extern/sentencepiece/src
                    extern/sentencepiece/src/third_party/absl/strings)

add_subdirectory(extern)
add_subdirectory(src)
