cmake_minimum_required(VERSION 3.21.1)

project(LLMRemoteServer VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

include(cmake/llama_setup.cmake)
include(cmake/download_llama_model.cmake)

find_package(Qt6 6.8 REQUIRED COMPONENTS Core RemoteObjects Concurrent WebSockets)

if (Qt6_VERSION VERSION_GREATER_EQUAL 6.3)
    qt_standard_project_setup()
endif()

qt_add_executable(LLMRemoteServer
    main.cpp
    InferenceEngine.h InferenceEngine.cpp
    QtRoRemoteGenerator.h QtRoRemoteGenerator.cpp
    QtWSRemoteGenerator.h QtWSRemoteGenerator.cpp
    ClientHandler.h ClientHandler.cpp
)

# ----------------------------------------------------------------------------
# C++コードからダウンロード済みの gguf モデルファイル名を参照できるようにする
# Enable referencing the downloaded gguf model file name from C++ code
# ----------------------------------------------------------------------------
target_compile_definitions(LLMRemoteServer PRIVATE
    LLAMA_MODEL_FILE="${LLAMA_MODEL_NAME}"
)

qt6_add_repc_sources(LLMRemoteServer
    ${CMAKE_CURRENT_LIST_DIR}/QtRemoteObjectsFiles/LlamaResponseGenerator.rep
)

find_library(LLAMA_LIB
    NAMES llama
    PATHS "${LLAMA_LIB_FILE_DIR}"
    NO_DEFAULT_PATH
    # 追加のパスがあれば追記 (Add extra paths here if necessary)
)

find_library(GGML_LIB
    NAMES ggml
    PATHS "${GGML_LIB_FILE_DIR}"
    NO_DEFAULT_PATH
    # 追加のパスがあれば追記 (Add extra paths here if necessary)
)

# 見つからない場合はビルドを中断
# Abort build if not found
if(LLAMA_LIB)
    message(STATUS "Found llama library: ${LLAMA_LIB}")
else()
    message(FATAL_ERROR "Could not find llama library (e.g. llama.lib / libllama.dylib / libllama.so)")
endif()

if(GGML_LIB)
    message(STATUS "Found ggml library: ${GGML_LIB}")
else()
    message(FATAL_ERROR "Could not find ggml library (e.g. ggml.lib / libggml.dylib / libggml.so)")
endif()

set(ALL_LIBS
    ${LLAMA_LIB}
    ${GGML_LIB}
)

target_include_directories(LLMRemoteServer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llama.cpp/ggml/include
)

if(APPLE)
    if(IOS)
        message(FATAL_ERROR "iOS is not supported; only macOS is supported")
    else()
        # macOS: .dylib をコピー
        # macOS: copy .dylib files
        add_custom_command(TARGET LLMRemoteServer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${LLAMA_DYNAMIC_LIB_FILE_DIR}/libllama.dylib"
            "$<TARGET_FILE_DIR:LLMRemoteServer>"
            COMMENT "Copying libllama.dylib to QllamaTalkApp"
        )

    file(GLOB GGML_DYLIBS
        "${GGML_DYNAMIC_LIB_FILE_DIR}/libggml*.dylib"
        "${GGML_DYNAMIC_LIB_FILE_DIR}/ggml-blas/libggml*.dylib"
        "${GGML_DYNAMIC_LIB_FILE_DIR}/ggml-metal/libggml*.dylib"
    )
foreach(dylib_file ${GGML_DYLIBS})
    add_custom_command(TARGET LLMRemoteServer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${dylib_file}"
        "$<TARGET_FILE_DIR:LLMRemoteServer>"
        COMMENT "Copying libggml*.dylib to QllamaTalkApp"
    )
endforeach()
endif()
else()
    message(FATAL_ERROR "only macOS is supported")
endif()

target_link_libraries(LLMRemoteServer PRIVATE
    Qt6::Core
    Qt6::RemoteObjects
    Qt6::Concurrent
    Qt6::WebSockets
    ${LLAMA_LIB}
    ${GGML_LIB}
)

# ----------------------------------------------------------------------------
# ダウンロードした gguf モデルファイルをアプリ実行ファイルの横にコピー
# Copy the downloaded gguf model file next to the app executable
# ----------------------------------------------------------------------------
add_custom_command(
    TARGET LLMRemoteServer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_LIST_DIR}/llama_models/${LLAMA_MODEL_NAME}"
    "$<TARGET_FILE_DIR:LLMRemoteServer>"
    COMMENT "Copying ${LLAMA_MODEL_NAME} next to the QllamaTalkApp binary"
)

include(GNUInstallDirs)
install(TARGETS LLMRemoteServer
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
