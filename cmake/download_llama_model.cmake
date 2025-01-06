#
# download_llama_model.cmake
#
# Always download the 8B model and place it in
# CMAKE_CURRENT_SOURCE_DIR/content/llama_models
# 常に8Bモデルをダウンロードし、
# CMAKE_CURRENT_SOURCE_DIR/content/llama_models に配置する

cmake_minimum_required(VERSION 3.16)

# Use the 8B model
# 8Bモデルを使用
set(LLAMA_MODEL_URL "https://huggingface.co/Triangle104/Llama-3.1-8B-Open-SFT-Q4_K_M-GGUF/resolve/main/llama-3.1-8b-open-sft-q4_k_m.gguf?download=true")
set(LLAMA_MODEL_NAME "llama-3.1-8b-open-sft-q4_k_m.gguf")

# Set destination path
# ダウンロード先のパス設定
set(LLAMA_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama_models")
set(LLAMA_MODEL_OUTPUT_PATH "${LLAMA_MODEL_DIR}/${LLAMA_MODEL_NAME}")

message(STATUS "----- Llama Model Download Setup -----")
message(STATUS "Model URL     : ${LLAMA_MODEL_URL}")
message(STATUS "Output Folder : ${LLAMA_MODEL_DIR}")
message(STATUS "Output File   : ${LLAMA_MODEL_OUTPUT_PATH}")
message(STATUS "--------------------------------------")

# Create directory if needed
# 必要に応じてディレクトリを作成
file(MAKE_DIRECTORY "${LLAMA_MODEL_DIR}")

# Check if file already exists
# ファイルが既に存在するか確認
if(EXISTS "${LLAMA_MODEL_OUTPUT_PATH}")
    message(STATUS "Llama model file already exists at: ${LLAMA_MODEL_OUTPUT_PATH}")
else()
    message(STATUS "Downloading Llama model from: ${LLAMA_MODEL_URL}")

    # Download the model
    # モデルをダウンロード
    file(DOWNLOAD
        "${LLAMA_MODEL_URL}"
        "${LLAMA_MODEL_OUTPUT_PATH}"
        SHOW_PROGRESS
        STATUS DOWNLOAD_STATUS
    )

    list(GET DOWNLOAD_STATUS 0 DOWNLOAD_RESULT_CODE)
    list(GET DOWNLOAD_STATUS 1 DOWNLOAD_ERROR_MESSAGE)

    # Check download result
    # ダウンロード結果を確認
    if(NOT DOWNLOAD_RESULT_CODE EQUAL 0)
        message(FATAL_ERROR "Failed to download Llama model. Error: ${DOWNLOAD_ERROR_MESSAGE}")
    else()
        message(STATUS "Llama model downloaded successfully.")
    endif()
endif()
