find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(EXT_LIBS ${EXT_LIBS} sentencepiece sentencepiece_train ${TORCH_LIBRARIES} ${CMAKE_DL_LIBS})

# Unnecessary?: include_directories(${CMAKE_CURRENT_BINARY_DIR})

# add_library(mtness STATIC
#   mtness.cpp
# )
# target_compile_options(mtness PUBLIC ${ALL_WARNINGS})

# set_target_properties(mtness PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
# set_target_properties(mtness PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

set(SRC_FILES mtness.cpp cli_options.cpp dataset.cpp
  models/sutskever.cpp models/att_decoder.cpp models/rnn.cpp
  models/encoder.cpp)

add_executable(mtness_main ${SRC_FILES})
set_target_properties(mtness_main PROPERTIES OUTPUT_NAME mtness)
target_compile_options(mtness_main PUBLIC ${ALL_WARNINGS})

set(EXECUTABLES ${EXECUTABLES} mtness_main)

foreach(exec ${EXECUTABLES})
  # target_link_libraries(${exec} mtness ${EXT_LIBS} ${CMAKE_THREAD_LIBS_INIT})
  target_link_libraries(${exec} ${EXT_LIBS} ${CMAKE_THREAD_LIBS_INIT})
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
endforeach(exec)
