cmake_minimum_required(VERSION 2.8)
project(game_of_life)

find_package(CUDA)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW )

# Pass options to NVCC
if (${CUDA_FOUND})
  set(CUDA_NVCC_FLAGS "-arch=sm_20")
endif()


if(${OPENGL_FOUND})
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
    
  add_library(${PROJECT_NAME}_serial
    src/serial.cpp)
  

  if(${CUDA_FOUND})
    cuda_add_library(${PROJECT_NAME}_simple_cuda_lib
      src/simple_cuda_gol.cu)
    target_link_libraries(${PROJECT_NAME}_simple_cuda_lib
      ${PROJECT_NAME}_utils
      ${OPENGL_LIBRARIES}
      ${GLUT_LIBRARY})
  endif()

  add_library(${PROJECT_NAME}_serial_lib
    src/game_of_life.cpp
    )
  target_link_libraries(${PROJECT_NAME}_serial_lib
    ${PROJECT_NAME}_utils
    ${PROJECT_NAME}_serial
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARY}
    ${GLEW_LIBRARY}
    )
  if(${CUDA_FOUND})
    cuda_add_executable(${PROJECT_NAME}.exe
      src/cuda_main.cpp)
    target_link_libraries(${PROJECT_NAME}.exe
      ${PROJECT_NAME}_serial_lib
      ${PROJECT_NAME}_simple_cuda_lib)
  else()
    add_executable(${PROJECT_NAME}.exe
      src/main.cpp)
    target_link_libraries(${PROJECT_NAME}.exe
      ${PROJECT_NAME}_serial_lib
      )
    add_executable(test.exe
      src/test.cpp)
    target_link_libraries(test.exe
      ${PROJECT_NAME}_serial
      )
  endif()

endif()




