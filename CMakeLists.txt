cmake_minimum_required(VERSION 3.10)
project(legged_wbc VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(-Wno-return-type -Wno-deprecated-declarations)

##############################
# 1️⃣ ROS1 / Catkin 支持
##############################
find_package(catkin QUIET)
if(catkin_FOUND)
  message(STATUS "Found catkin")
  catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
  )
endif()

##############################
# 2️⃣ 纯 C++ 依赖
##############################
find_package(qpOASES REQUIRED)
find_package(osqp REQUIRED)
find_package(OsqpEigen REQUIRED)
find_package(legged_model REQUIRED)

##############################
# 3️⃣ 创建库（PUBLIC 导出依赖）
##############################
add_library(${PROJECT_NAME} STATIC
  src/WbcBase.cpp
  src/WeightedWbc.cpp
)

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# 对下游暴露你自己的头文件
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# qpOASES：有的安装只给变量，没有 target。这里做兼容
if(TARGET qpOASES::qpOASES)
  target_link_libraries(${PROJECT_NAME} PUBLIC qpOASES::qpOASES)
elseif(DEFINED qpOASES_LIBRARY)
  target_include_directories(${PROJECT_NAME} PUBLIC ${qpOASES_INCLUDE_DIR})
  target_link_libraries(${PROJECT_NAME} PUBLIC ${qpOASES_LIBRARY})
else()
  # 兜底：如果包只给 qpOASES_LIBRARIES 之类变量，你可按实际变量名改
  target_include_directories(${PROJECT_NAME} PUBLIC ${qpOASES_INCLUDE_DIR})
  target_link_libraries(${PROJECT_NAME} PUBLIC ${qpOASES_LIBRARY})
endif()

# OSQP / OsqpEigen：通常是标准 target
target_link_libraries(${PROJECT_NAME} PUBLIC
  osqp::osqp
  OsqpEigen::OsqpEigen
  legged_model::legged_model
)

# 一般不需要硬编码 /usr/local/include/osqp
# 如果你当前 osqp 包没有正确导出 include（极少数情况），再打开下面这行：
# target_include_directories(${PROJECT_NAME} PUBLIC /usr/local/include/osqp)

##############################
# 4️⃣ 可执行文件（本地测试）
##############################
add_executable(WeightedWbc_test test/WeightedWbc_test.cpp)
target_link_libraries(WeightedWbc_test PRIVATE ${PROJECT_NAME}::${PROJECT_NAME})

##############################
# 5️⃣ 安装：库 + 头文件 + CMake Package
##############################
install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

##############################
# 6️⃣ ROS1 安装（可选）
##############################
if(catkin_FOUND)
  install(TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
  install(DIRECTORY include/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  )
endif()

##############################
# 7️⃣ ROS2 / ament_cmake 支持（导出依赖）
##############################
find_package(ament_cmake QUIET)
if(ament_cmake_FOUND)
  message(STATUS "Found ament_cmake")

  install(TARGETS WeightedWbc_test
    DESTINATION lib/${PROJECT_NAME}
  )

  ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
  ament_export_dependencies(qpOASES osqp OsqpEigen legged_model)

  ament_package()
endif()
