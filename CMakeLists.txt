cmake_minimum_required(VERSION 3.10)
project(legged_wbc)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(catkin QUIET)
if(catkin_FOUND)
    message("Found catkin")
    catkin_package(
        INCLUDE_DIRS include
        LIBRARIES ${PROJECT_NAME}
    )
endif()

##############################
# 1️⃣ 与 ROS 无关的部分（纯 C++）
##############################

# 第三方库
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(qpOASES REQUIRED)
find_package(pinocchio REQUIRED)

# include 目录
include_directories(
    include
    ${EIGEN3_INCLUDE_DIRS}
    ${pinocchio_INCLUDE_DIRS}
    ${qpOASES_INCLUDE_DIR}
)

# library 搜索路径
link_directories(
    ${pinocchio_LIBRARY_DIRS}
)

# 库
add_library(${PROJECT_NAME}
    src/HoQp.cpp
    src/WbcBase.cpp
    src/HierarchicalWbc.cpp
    src/WeightedWbc.cpp
    src/LeggedState.cpp
    src/LeggedModel.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${pinocchio_LIBRARIES}
    ${qpOASES_LIBRARY}
    yaml-cpp
)

# 可执行文件示例（独立测试）
add_executable(LeggedState_test
    test/LeggedState_test.cpp
    src/LeggedState.cpp
)
target_link_libraries(LeggedState_test ${pinocchio_LIBRARIES})

add_executable(LeggedModel_test
    test/LeggedModel_test.cpp
    src/LeggedState.cpp
    src/LeggedModel.cpp
)
target_link_libraries(LeggedModel_test ${pinocchio_LIBRARIES})

add_executable(WeightedWbc_test
    test/WeightedWbc_test.cpp
    src/LeggedState.cpp
    src/LeggedModel.cpp
    src/WbcBase.cpp
    src/WeightedWbc.cpp
)
target_link_libraries(WeightedWbc_test ${pinocchio_LIBRARIES} ${qpOASES_LIBRARY} yaml-cpp)

##############################
# 2️⃣ ROS 适配部分（可选）
##############################

# ROS1
if(catkin_FOUND)
    # 安装库和头文件
    install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )
    install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
    )

    # ROS 测试
    if (CATKIN_ENABLE_TESTING)
        catkin_add_gtest(${PROJECT_NAME}_test
            test/HoQp_test.cpp
        )
        target_link_libraries(${PROJECT_NAME}_test
            ${PROJECT_NAME}
            ${pinocchio_LIBRARIES}
            gtest_main
        )
    endif()
endif()

# ROS2
find_package(ament_cmake QUIET)
if(ament_cmake_FOUND)
    ament_package(
        INCLUDE_DIRS include
        LIBRARIES ${PROJECT_NAME}
    )

    install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
    install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION include
    )
endif()
