cmake_minimum_required(VERSION 3.21.1)

project(LLMServerQtRemoteObjects LANGUAGES CXX)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 6.8 REQUIRED COMPONENTS Core)
find_package(Qt6 REQUIRED COMPONENTS RemoteObjects)

if (Qt6_VERSION VERSION_GREATER_EQUAL 6.3)
    qt_standard_project_setup()
endif()

qt_add_executable(LLMServerQtRemoteObjects
    main.cpp
    llamaresponsegenerator.cpp
    llamaresponsegenerator.h
)

qt6_add_repc_sources(LLMServerQtRemoteObjects
    ${CMAKE_CURRENT_SOURCE_DIR}/QtRemoteObjectsFiles/LlamaResponseGenerator.rep
)

target_link_libraries(LLMServerQtRemoteObjects PRIVATE
    Qt6::Core
    Qt6::RemoteObjects
)
include(GNUInstallDirs)
install(TARGETS LLMServerQtRemoteObjects
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
