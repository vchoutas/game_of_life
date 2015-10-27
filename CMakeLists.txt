cmake_minimum_required(VERSION 2.8)
project(game_of_life)

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

# Pass options to NVCC
set(CUDA_NVCC_FLAGS "-arch=sm_20")


set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

include_directories(
  include
  ${OPENGL_INCLUDE_DIRS}
  ${GLUT_INCLUDE_DIRS}
  ${GLEW_INCLUDE_DIRS}
  )

add_library(${PROJECT_NAME}_utils
  src/utilities.cpp)
target_link_libraries(${PROJECT_NAME}_utils
  ${OPENGL_LIBRARIES})


cuda_add_library(${PROJECT_NAME}_kernels
  src/cuda_kernels.cu)
target_link_libraries(${PROJECT_NAME}_kernels
  ${PROJECT_NAME}_utils
  )
cuda_add_library(${PROJECT_NAME}_cuda
  src/game_of_life.cu)
target_link_libraries(${PROJECT_NAME}_cuda
  ${PROJECT_NAME}_kernels
  ${PROJECT_NAME}_utils
  ${OPENGL_LIBRARIES}
  ${GLUT_LIBRARY}
  ${GLEW_LIBRARY}
  )



cuda_add_executable(${PROJECT_NAME}.exe
  src/main.cpp)
target_link_libraries(${PROJECT_NAME}.exe
  ${PROJECT_NAME}_cuda)





