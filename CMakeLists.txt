cmake_minimum_required(VERSION 3.10)
project(legged_wbc VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

##############################
# 1️⃣ ROS1 / Catkin 支持
##############################
find_package(catkin QUIET)
if(catkin_FOUND)
    message("Found catkin")
    catkin_package(
        INCLUDE_DIRS include
        LIBRARIES ${PROJECT_NAME}
    )
endif()

##############################
# 2️⃣ 与 ROS 无关的部分（纯 C++）
##############################

# 第三方库
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(qpOASES REQUIRED)
find_package(logger REQUIRED)

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

# 创建库
add_library(${PROJECT_NAME} STATIC
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
    logger::CsvLogger
)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIRS}
    ${pinocchio_INCLUDE_DIRS}
    ${qpOASES_INCLUDE_DIR}
)

##############################
# 3️⃣ 安装配置
##############################

# 安装库
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# 安装头文件
install(DIRECTORY include/ DESTINATION include)

# 安装导出文件
install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Config.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/cmake/${PROJECT_NAME}
)

##############################
# 4️⃣ 可执行文件（独立测试）
##############################

add_executable(LeggedState_test
    test/LeggedState_test.cpp
)
target_link_libraries(LeggedState_test ${PROJECT_NAME} ${pinocchio_LIBRARIES})

add_executable(LeggedModel_test
    test/LeggedModel_test.cpp
    src/LeggedModel.cpp
)
target_link_libraries(LeggedModel_test ${PROJECT_NAME} ${pinocchio_LIBRARIES} yaml-cpp)

add_executable(WeightedWbc_test
    test/WeightedWbc_test.cpp
    src/LeggedModel.cpp
    src/WbcBase.cpp
    src/WeightedWbc.cpp
)
target_link_libraries(WeightedWbc_test ${PROJECT_NAME} ${pinocchio_LIBRARIES} ${qpOASES_LIBRARY} yaml-cpp)

##############################
# 5️⃣ ROS1 测试与安装
##############################

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

    # ROS1 测试
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

##############################
# 6️⃣ ROS2 / ament_cmake 支持
##############################

find_package(ament_cmake QUIET)
if(ament_cmake_FOUND)
    message("Found ament_cmake")

    # 安装可执行文件
    install(TARGETS LeggedState_test LeggedModel_test WeightedWbc_test
        DESTINATION lib/${PROJECT_NAME}
    )

    ament_package()
endif()
