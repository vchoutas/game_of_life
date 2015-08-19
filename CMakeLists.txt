cmake_minimum_required(VERSION 2.8)
project(game_of_life)

find_package(CUDA QUIET REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

# Pass options to NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_22,code=sm_22)


if(${OPENGL_FOUND} AND ${CUDA_FOUND})
  set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

  include_directories(
    include
    ${OPENGL_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
    )

  add_library(${PROJECT_NAME}_utils
    src/utilities.cpp)
  target_link_libraries(${PROJECT_NAME}_utils
    ${OPENGL_LIBRARIES})

  add_library(${PROJECT_NAME}_serial_lib
    src/game_of_life.cpp
    )
  target_link_libraries(${PROJECT_NAME}_serial_lib
    ${PROJECT_NAME}_utils
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARY})

  add_executable(${PROJECT_NAME}_serial.exe
    src/main.cpp)
  target_link_libraries(${PROJECT_NAME}_serial.exe
    ${PROJECT_NAME}_serial_lib)

endif()




