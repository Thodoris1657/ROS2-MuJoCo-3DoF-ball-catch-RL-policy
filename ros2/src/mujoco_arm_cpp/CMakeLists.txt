cmake_minimum_required(VERSION 3.16)
project(mujoco_arm_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)
# GCC false positives inside nlohmann/json's iterators
add_compile_options(-Wno-maybe-uninitialized -Wno-uninitialized)

# ---------------------------------------------------------------- MuJoCo
# Default: the MuJoCo shipped inside the `mujoco` pip wheel of the active Python, so the C++
# nodes run *exactly* the physics library the Python training / benchmarks used.
# Override with -DMUJOCO_INCLUDE_DIR=... -DMUJOCO_LIBRARY=... (e.g. a release tarball).
if(NOT MUJOCO_INCLUDE_DIR OR NOT MUJOCO_LIBRARY)
  find_package(Python3 COMPONENTS Interpreter)
  if(DEFINED ENV{MUJOCO_PYTHON})
    set(_mj_py $ENV{MUJOCO_PYTHON})
  else()
    set(_mj_py ${Python3_EXECUTABLE})
  endif()
  execute_process(
    COMMAND ${_mj_py} -c "import mujoco, os; print(os.path.dirname(mujoco.__file__))"
    OUTPUT_VARIABLE _mj_dir OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _mj_rc)
  if(NOT _mj_rc EQUAL 0)
    message(FATAL_ERROR "MuJoCo not found: `pip install mujoco` in the active Python (or set "
                        "MUJOCO_PYTHON / MUJOCO_INCLUDE_DIR + MUJOCO_LIBRARY)")
  endif()
  set(MUJOCO_INCLUDE_DIR ${_mj_dir}/include)
  file(GLOB _mj_libs ${_mj_dir}/libmujoco.so*)
  list(GET _mj_libs 0 MUJOCO_LIBRARY)
endif()
message(STATUS "MuJoCo headers: ${MUJOCO_INCLUDE_DIR}")
message(STATUS "MuJoCo library: ${MUJOCO_LIBRARY}")
get_filename_component(MUJOCO_LIB_DIR ${MUJOCO_LIBRARY} DIRECTORY)

find_package(nlohmann_json 3.2 REQUIRED)

# The repo's assets / configs / benchmarks (this package lives in <repo>/ros2/src/mujoco_arm_cpp)
get_filename_component(REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../.. ABSOLUTE)

# ---------------------------------------------------------- core library
add_library(mujoco_arm_core
  src/arm_model.cpp src/catch_sim.cpp src/catch_controller.cpp src/reach_sim.cpp src/policies.cpp
  src/catch_core.cpp)
target_include_directories(mujoco_arm_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
  ${MUJOCO_INCLUDE_DIR})
target_link_libraries(mujoco_arm_core PUBLIC ${MUJOCO_LIBRARY} nlohmann_json::nlohmann_json)
set_target_properties(mujoco_arm_core PROPERTIES POSITION_INDEPENDENT_CODE ON
  BUILD_RPATH "${MUJOCO_LIB_DIR}" INSTALL_RPATH "${MUJOCO_LIB_DIR}")

# ROS-free command-line benchmark
add_executable(catch_bench src/tools/catch_bench.cpp)
target_link_libraries(catch_bench mujoco_arm_core)
set_target_properties(catch_bench PROPERTIES BUILD_RPATH "${MUJOCO_LIB_DIR}" INSTALL_RPATH "${MUJOCO_LIB_DIR}")

add_executable(catch_eval_params src/tools/catch_eval_params.cpp)
target_link_libraries(catch_eval_params mujoco_arm_core)
set_target_properties(catch_eval_params PROPERTIES BUILD_RPATH "${MUJOCO_LIB_DIR}" INSTALL_RPATH "${MUJOCO_LIB_DIR}")

# ------------------------------------------------------------ core tests
# Plain GoogleTest: runs without ROS (`cmake -S . -B build && ctest --test-dir build`).
option(MUJOCO_ARM_TESTS "build the core library tests" ON)
if(MUJOCO_ARM_TESTS)
  find_package(GTest QUIET)
  if(GTest_FOUND)
    enable_testing()
    add_executable(test_core test/test_core.cpp test/test_parity.cpp)
    target_link_libraries(test_core mujoco_arm_core GTest::gtest_main)
    target_compile_definitions(test_core PRIVATE
      REPO_ROOT="${REPO_ROOT}" GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    set_target_properties(test_core PROPERTIES BUILD_RPATH "${MUJOCO_LIB_DIR}")
    add_test(NAME test_core COMMAND test_core)
  else()
    message(STATUS "GTest not found: core tests skipped (apt install libgtest-dev)")
  endif()
endif()

# ------------------------------------------------------------- ROS nodes
find_package(ament_cmake QUIET)
if(NOT ament_cmake_FOUND)
  message(STATUS "ament_cmake not found: building the ROS-free core only")
  return()
endif()

find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(mujoco_arm_interfaces REQUIRED)

set(NODE_DEPS rclcpp rclcpp_action geometry_msgs sensor_msgs std_msgs std_srvs
  visualization_msgs tf2_ros rcl_interfaces mujoco_arm_interfaces)

add_library(mujoco_arm_ros_common src/ros_common.cpp)
target_include_directories(mujoco_arm_ros_common PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(mujoco_arm_ros_common PUBLIC mujoco_arm_core)
target_compile_definitions(mujoco_arm_ros_common PRIVATE PKG_NAME="${PROJECT_NAME}")
foreach(dep ${NODE_DEPS})
  target_link_libraries(mujoco_arm_ros_common PUBLIC ${${dep}_TARGETS})
endforeach()
find_package(ament_index_cpp REQUIRED)
target_link_libraries(mujoco_arm_ros_common PUBLIC ament_index_cpp::ament_index_cpp)

set(NODES sim_node policy_node reach_server eval_client demo_targets
          catch_sim_node catch_server catch_demo catch_eval_client)
foreach(node ${NODES})
  add_executable(${node} src/nodes/${node}.cpp)
  target_link_libraries(${node} mujoco_arm_ros_common)
  set_target_properties(${node} PROPERTIES BUILD_RPATH "${MUJOCO_LIB_DIR}"
    INSTALL_RPATH "${MUJOCO_LIB_DIR};${CMAKE_INSTALL_PREFIX}/lib")
  install(TARGETS ${node} DESTINATION lib/${PROJECT_NAME})
endforeach()
install(TARGETS catch_bench catch_eval_params DESTINATION lib/${PROJECT_NAME})

install(TARGETS mujoco_arm_core mujoco_arm_ros_common
  LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY launch config urdf rviz DESTINATION share/${PROJECT_NAME})
# self-contained share/: the MJCF models, tuned parameters and frozen benchmark sets
install(FILES ${REPO_ROOT}/assets/arm3.xml ${REPO_ROOT}/assets/arm3_ballcatch.xml
  DESTINATION share/${PROJECT_NAME}/assets)
install(FILES ${REPO_ROOT}/configs/catch_tuned.json DESTINATION share/${PROJECT_NAME}/config)
install(FILES ${REPO_ROOT}/benchmarks/eval_set_v1.json ${REPO_ROOT}/benchmarks/catch_set_v1.json
  ${REPO_ROOT}/benchmarks/catch_set_v1_stress.json DESTINATION share/${PROJECT_NAME}/benchmarks)

if(BUILD_TESTING AND TARGET test_core)
  find_package(ament_cmake_gtest QUIET)
endif()

ament_package()
