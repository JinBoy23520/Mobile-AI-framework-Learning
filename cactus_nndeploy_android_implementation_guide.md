# Cactus-nndeploy Android 端详细实现指南

## 📋 文档说明

本文档针对**已成功集成 Cactus Android 端源码并运行 GGUF 模型**的开发者，提供深度的工作流集成、优化和生产部署指导。

**前置条件**：
- ✅ Cactus Android 端已编译成功
- ✅ GGUF 模型可以正常加载和推理
- ✅ 熟悉 nndeploy 框架基础
- ✅ 了解 Android NDK 开发

---

## 📑 目录

### Part 1: Cactus 源码集成详解
- [1.1 Cactus 源码结构分析](#11-cactus-源码结构分析)
- [1.2 Android 编译配置](#12-android-编译配置)
- [1.3 JNI 桥接层实现](#13-jni-桥接层实现)

### Part 2: nndeploy 后端适配
- [2.1 Backend Adapter 完整实现](#21-backend-adapter-完整实现)
- [2.2 GGUF 模型加载器详解](#22-gguf-模型加载器详解)
- [2.3 内存管理和优化](#23-内存管理和优化)

### Part 3: 工作流集成设计
- [3.1 DAG 节点封装](#31-dag-节点封装)
- [3.2 LLM 工作流模式](#32-llm-工作流模式)
- [3.3 多模态工作流](#33-多模态工作流)

### Part 4: 实战案例
- [4.1 RAG 工作流实现](#41-rag-工作流实现)
- [4.2 Agent 工作流实现](#42-agent-工作流实现)
- [4.3 流式推理工作流](#43-流式推理工作流)

---

## Part 1: Cactus 源码集成详解

### 1.1 Cactus 源码结构分析

#### 核心目录结构

```
third_party/cactus/
├── cactus/
│   ├── engine/                    # 核心推理引擎
│   │   ├── engine.h              # 主引擎接口
│   │   ├── engine_model.cpp      # 模型基类实现
│   │   ├── engine_config.cpp     # 配置管理
│   │   └── engine_tokenizer.cpp  # 分词器
│   │
│   ├── models/                    # 模型实现
│   │   ├── model.h               # 模型抽象接口
│   │   ├── model_gemma.cpp       # Gemma 模型
│   │   ├── model_llama.cpp       # Llama 模型
│   │   ├── model_qwen.cpp        # Qwen 模型
│   │   ├── model_lfm2.cpp        # LFM2 模型
│   │   ├── model_lfm2vl.cpp      # LFM2 VLM
│   │   ├── model_whisper.cpp     # Whisper 语音
│   │   └── model_siglip2.cpp     # SigLIP2 视觉
│   │
│   ├── graph/                     # 计算图
│   │   ├── graph.h               # 图构建接口
│   │   ├── graph.cpp             # 图执行引擎
│   │   ├── graph_buffer.cpp      # 缓冲区管理
│   │   └── graph_ops.cpp         # 算子实现
│   │
│   ├── ffi/                       # C FFI 接口
│   │   ├── cactus_ffi.h          # FFI 头文件
│   │   ├── cactus_init.cpp       # 初始化接口
│   │   ├── cactus_complete.cpp   # 补全接口
│   │   ├── cactus_transcribe.cpp # 转录接口
│   │   └── cactus_utils.h        # 工具函数
│   │
│   ├── npu/                       # NPU 加速（可选）
│   │   ├── npu.h                 # NPU 接口
│   │   ├── npu_apple.mm          # Apple ANE
│   │   └── npu_qualcomm.cpp      # Qualcomm HTP
│   │
│   └── kernels/                   # 计算核心
│       ├── cpu/                  # CPU 实现
│       │   ├── matmul_int8.cpp   # INT8 矩阵乘法
│       │   ├── attention.cpp     # 注意力机制
│       │   └── rope.cpp          # RoPE 位置编码
│       └── neon/                 # ARM NEON 优化
│           └── matmul_neon.cpp
│
└── CMakeLists.txt                 # Android 构建配置
```

#### Android 特定代码位置

```
third_party/cactus/android/
├── app/                          # Android 示例应用
│   └── src/main/
│       ├── java/                 # Java/Kotlin 代码
│       └── cpp/                  # JNI 实现
│
└── libs/                         # 预编译库（如果有）
```

#### 关键文件功能说明

| 文件 | 功能 | Android 集成重要性 |
|------|------|------------------|
| `cactus/ffi/cactus_ffi.h` | C FFI 接口定义 | ⭐⭐⭐⭐⭐ 必须 |
| `cactus/engine/engine_model.cpp` | 模型加载和推理 | ⭐⭐⭐⭐⭐ 必须 |
| `cactus/graph/graph.cpp` | 计算图执行 | ⭐⭐⭐⭐⭐ 必须 |
| `cactus/models/model_*.cpp` | 具体模型实现 | ⭐⭐⭐⭐ 可选（按需） |
| `cactus/npu/npu_*.cpp` | NPU 加速 | ⭐⭐⭐ 可选 |
| `cactus/kernels/neon/` | NEON 优化 | ⭐⭐⭐⭐ 推荐 |

---

### 1.2 Android 编译配置

#### CMakeLists.txt 详细配置

创建 `third_party/cactus/CMakeLists.txt.android`：

```cmake
# Cactus Android 构建配置
cmake_minimum_required(VERSION 3.19)
project(cactus_android CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Android 特定配置
if(ANDROID)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math -DNDEBUG")
    
    # ARM NEON 支持
    if(ANDROID_ABI STREQUAL "armeabi-v7a")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -mfloat-abi=softfp")
    elseif(ANDROID_ABI STREQUAL "arm64-v8a")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+fp+simd")
    endif()
endif()

# ============================================
# Cactus 核心库
# ============================================

# 收集源文件
file(GLOB_RECURSE CACTUS_ENGINE_SOURCES
    "cactus/engine/*.cpp"
)

file(GLOB_RECURSE CACTUS_GRAPH_SOURCES
    "cactus/graph/*.cpp"
)

file(GLOB_RECURSE CACTUS_FFI_SOURCES
    "cactus/ffi/*.cpp"
)

# 模型源文件（按需选择）
set(CACTUS_MODEL_SOURCES
    cactus/models/model_gemma.cpp
    cactus/models/model_qwen.cpp
    cactus/models/model_lfm2.cpp
    cactus/models/model_smol.cpp
    # cactus/models/model_lfm2vl.cpp    # VLM 支持
    # cactus/models/model_whisper.cpp   # 语音支持
    # cactus/models/model_siglip2.cpp   # 视觉编码器
)

# CPU 内核
file(GLOB CACTUS_KERNEL_CPU_SOURCES
    "cactus/kernels/cpu/*.cpp"
)

# NEON 优化内核
if(ANDROID_ABI MATCHES "^arm")
    file(GLOB CACTUS_KERNEL_NEON_SOURCES
        "cactus/kernels/neon/*.cpp"
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_NEON=1")
endif()

# NPU 支持（可选）
option(ENABLE_CACTUS_NPU "Enable NPU acceleration" OFF)
if(ENABLE_CACTUS_NPU AND ANDROID_ABI STREQUAL "arm64-v8a")
    # Qualcomm HTP 或其他 NPU 支持
    set(CACTUS_NPU_SOURCES cactus/npu/npu_qualcomm.cpp)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_NPU=1")
endif()

# 组合所有源文件
set(CACTUS_ALL_SOURCES
    ${CACTUS_ENGINE_SOURCES}
    ${CACTUS_GRAPH_SOURCES}
    ${CACTUS_FFI_SOURCES}
    ${CACTUS_MODEL_SOURCES}
    ${CACTUS_KERNEL_CPU_SOURCES}
    ${CACTUS_KERNEL_NEON_SOURCES}
    ${CACTUS_NPU_SOURCES}
)

# 创建共享库
add_library(cactus_engine SHARED ${CACTUS_ALL_SOURCES})

# 头文件路径
target_include_directories(cactus_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cactus
)

# 链接 Android 系统库
target_link_libraries(cactus_engine
    log           # Android 日志
    android       # Android API
)

# 导出符号
set_target_properties(cactus_engine PROPERTIES
    PUBLIC_HEADER "cactus/ffi/cactus_ffi.h"
    VERSION 1.3.0
    SOVERSION 1
)

# 安装配置
install(TARGETS cactus_engine
    LIBRARY DESTINATION lib/${ANDROID_ABI}
    PUBLIC_HEADER DESTINATION include/cactus
)

# ============================================
# 编译选项控制
# ============================================

# INT8 量化（默认启用）
option(ENABLE_INT8_QUANT "Enable INT8 quantization" ON)
if(ENABLE_INT8_QUANT)
    target_compile_definitions(cactus_engine PRIVATE ENABLE_INT8=1)
endif()

# FP16 支持
option(ENABLE_FP16 "Enable FP16 support" ON)
if(ENABLE_FP16 AND ANDROID_ABI STREQUAL "arm64-v8a")
    target_compile_definitions(cactus_engine PRIVATE ENABLE_FP16=1)
endif()

# 调试日志
option(ENABLE_DEBUG_LOG "Enable debug logging" OFF)
if(ENABLE_DEBUG_LOG)
    target_compile_definitions(cactus_engine PRIVATE CACTUS_DEBUG=1)
endif()

# 性能分析
option(ENABLE_PROFILING "Enable profiling" OFF)
if(ENABLE_PROFILING)
    target_compile_definitions(cactus_engine PRIVATE ENABLE_PROFILING=1)
endif()

# ============================================
# 打印配置信息
# ============================================

message(STATUS "========================================")
message(STATUS "Cactus Android Build Configuration")
message(STATUS "========================================")
message(STATUS "Android ABI:        ${ANDROID_ABI}")
message(STATUS "Build Type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard:       ${CMAKE_CXX_STANDARD}")
message(STATUS "INT8 Quant:         ${ENABLE_INT8_QUANT}")
message(STATUS "FP16 Support:       ${ENABLE_FP16}")
message(STATUS "NPU Acceleration:   ${ENABLE_CACTUS_NPU}")
message(STATUS "Debug Log:          ${ENABLE_DEBUG_LOG}")
message(STATUS "========================================")
```

#### nndeploy 集成配置

在 `nndeploy/cmake/config_android_cactus.cmake` 中添加：

```cmake
# ============================================
# Cactus Backend 配置
# ============================================

# 启用 Cactus 后端
set(ENABLE_NNDEPLOY_INFERENCE_CACTUS ON)

# Cactus 源码路径
set(CACTUS_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third_party/cactus")
set(CACTUS_INCLUDE_DIR "${CACTUS_SOURCE_DIR}")

# Cactus 编译选项
set(ENABLE_INT8_QUANT ON)
set(ENABLE_FP16 ON)
set(ENABLE_CACTUS_NPU OFF)  # 默认关闭，需要时开启
set(ENABLE_DEBUG_LOG OFF)

# 添加 Cactus 子项目
add_subdirectory(${CACTUS_SOURCE_DIR} cactus_build)

# ============================================
# nndeploy Cactus Backend 实现
# ============================================

# Backend 源文件
set(NNDEPLOY_CACTUS_BACKEND_SOURCES
    framework/source/nndeploy/inference/cactus/cactus_inference.cc
    framework/source/nndeploy/inference/cactus/cactus_backend.cc
    framework/source/nndeploy/inference/cactus/gguf_loader.cc
)

# 添加到 nndeploy framework
target_sources(nndeploy_framework PRIVATE
    ${NNDEPLOY_CACTUS_BACKEND_SOURCES}
)

# 包含路径
target_include_directories(nndeploy_framework PRIVATE
    ${CACTUS_INCLUDE_DIR}
)

# 链接 Cactus 库
target_link_libraries(nndeploy_framework PRIVATE
    cactus_engine
)

# 定义宏
target_compile_definitions(nndeploy_framework PRIVATE
    ENABLE_NNDEPLOY_CACTUS=1
)
```

#### 编译脚本

创建 `build_android_cactus.sh`：

```bash
#!/bin/bash
set -e

# ============================================
# Cactus-nndeploy Android 构建脚本
# ============================================

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================
# 配置参数
# ============================================

# Android NDK 路径（必须设置）
if [ -z "$ANDROID_NDK" ]; then
    echo_error "ANDROID_NDK environment variable not set"
    echo_info "Please set: export ANDROID_NDK=/path/to/android-ndk"
    exit 1
fi

# 架构选择
ANDROID_ABI=${ANDROID_ABI:-arm64-v8a}
ANDROID_PLATFORM=${ANDROID_PLATFORM:-android-24}
BUILD_TYPE=${BUILD_TYPE:-Release}

# 编译选项
ENABLE_NPU=${ENABLE_NPU:-OFF}
ENABLE_VLM=${ENABLE_VLM:-OFF}
ENABLE_WHISPER=${ENABLE_WHISPER:-OFF}
JOBS=${JOBS:-$(nproc)}

# 输出目录
BUILD_DIR="build_android_${ANDROID_ABI}"
INSTALL_DIR="${BUILD_DIR}/install"

echo_info "========================================="
echo_info "Build Configuration"
echo_info "========================================="
echo_info "NDK Path:        $ANDROID_NDK"
echo_info "ABI:             $ANDROID_ABI"
echo_info "Platform:        $ANDROID_PLATFORM"
echo_info "Build Type:      $BUILD_TYPE"
echo_info "NPU Support:     $ENABLE_NPU"
echo_info "VLM Support:     $ENABLE_VLM"
echo_info "Whisper:         $ENABLE_WHISPER"
echo_info "Parallel Jobs:   $JOBS"
echo_info "========================================="

# ============================================
# 清理（可选）
# ============================================

if [ "$1" = "--clean" ]; then
    echo_warn "Cleaning build directory: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

# ============================================
# 创建构建目录
# ============================================

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# ============================================
# CMake 配置
# ============================================

echo_info "Configuring CMake..."

cmake -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="$ANDROID_ABI" \
    -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
    -DANDROID_STL=c++_shared \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DCMAKE_CXX_FLAGS="-O3 -ffast-math" \
    -DENABLE_CACTUS_NPU="$ENABLE_NPU" \
    -DENABLE_INT8_QUANT=ON \
    -DENABLE_FP16=ON \
    -C ../cmake/config_android_cactus.cmake \
    ..

if [ $? -ne 0 ]; then
    echo_error "CMake configuration failed"
    exit 1
fi

# ============================================
# 编译
# ============================================

echo_info "Building with $JOBS parallel jobs..."

ninja -j"$JOBS"

if [ $? -ne 0 ]; then
    echo_error "Build failed"
    exit 1
fi

# ============================================
# 安装
# ============================================

echo_info "Installing to $INSTALL_DIR..."

ninja install

if [ $? -ne 0 ]; then
    echo_error "Installation failed"
    exit 1
fi

# ============================================
# 验证输出
# ============================================

echo_info "Verifying build outputs..."

if [ -f "$INSTALL_DIR/lib/$ANDROID_ABI/libcactus_engine.so" ]; then
    echo_info "✓ libcactus_engine.so"
else
    echo_error "✗ libcactus_engine.so not found"
    exit 1
fi

if [ -f "$INSTALL_DIR/lib/$ANDROID_ABI/libnndeploy_framework.so" ]; then
    echo_info "✓ libnndeploy_framework.so"
else
    echo_error "✗ libnndeploy_framework.so not found"
    exit 1
fi

# ============================================
# 完成
# ============================================

echo_info "========================================="
echo_info "Build completed successfully!"
echo_info "========================================="
echo_info "Libraries location:"
echo_info "  $INSTALL_DIR/lib/$ANDROID_ABI/"
echo_info ""
echo_info "To use in Android Studio:"
echo_info "  1. Copy libs to app/src/main/jniLibs/$ANDROID_ABI/"
echo_info "  2. Load library: System.loadLibrary(\"cactus_engine\")"
echo_info "========================================="

# 可选：自动复制到 Android 项目
if [ -n "$ANDROID_PROJECT_PATH" ]; then
    JNI_LIBS_DIR="$ANDROID_PROJECT_PATH/app/src/main/jniLibs/$ANDROID_ABI"
    echo_info "Copying libraries to Android project..."
    mkdir -p "$JNI_LIBS_DIR"
    cp "$INSTALL_DIR/lib/$ANDROID_ABI"/*.so "$JNI_LIBS_DIR/"
    echo_info "✓ Libraries copied to $JNI_LIBS_DIR"
fi
```

使用方法：

```bash
# 基础编译
export ANDROID_NDK=/path/to/ndk
./build_android_cactus.sh

# 清理重新编译
./build_android_cactus.sh --clean

# 启用 NPU 支持
ENABLE_NPU=ON ./build_android_cactus.sh

# 自动复制到 Android 项目
export ANDROID_PROJECT_PATH=/path/to/android/project
./build_android_cactus.sh
```

---

### 1.3 JNI 桥接层实现

#### JNI 接口定义

创建 `framework/source/nndeploy/inference/cactus/cactus_jni.cpp`：

```cpp
#include <jni.h>
#include <android/log.h>
#include <string>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "cactus/ffi/cactus_ffi.h"
#include "nndeploy/inference/cactus/cactus_inference.h"

#define LOG_TAG "CactusJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ============================================
// 全局模型管理
// ============================================

struct ModelHandle {
    cactus_model_t model;
    std::string model_path;
    size_t context_size;
    std::mutex mutex;
};

// 模型句柄映射 (model_id -> ModelHandle)
static std::unordered_map<int64_t, std::unique_ptr<ModelHandle>> g_models;
static std::mutex g_models_mutex;
static int64_t g_next_model_id = 1;

// ============================================
// 辅助函数
// ============================================

// Java String 转 C++ string
std::string jstring_to_string(JNIEnv* env, jstring jstr) {
    if (!jstr) return "";
    const char* chars = env->GetStringUTFChars(jstr, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return result;
}

// C++ string 转 Java String
jstring string_to_jstring(JNIEnv* env, const std::string& str) {
    return env->NewStringUTF(str.c_str());
}

// 获取模型句柄
ModelHandle* get_model_handle(int64_t model_id) {
    std::lock_guard<std::mutex> lock(g_models_mutex);
    auto it = g_models.find(model_id);
    return (it != g_models.end()) ? it->second.get() : nullptr;
}

// ============================================
// JNI 方法实现
// ============================================

extern "C" {

/**
 * 从 GGUF 文件初始化模型
 * 
 * @param ggufPath GGUF 模型文件路径
 * @param contextSize 上下文窗口大小
 * @return 模型 ID (>0 成功, <=0 失败)
 */
JNIEXPORT jlong JNICALL
Java_com_nndeploy_cactus_CactusEngine_nativeInitFromGGUF(
    JNIEnv* env,
    jobject /* this */,
    jstring ggufPath,
    jint contextSize
) {
    try {
        std::string model_path = jstring_to_string(env, ggufPath);
        
        LOGI("Initializing model from: %s", model_path.c_str());
        LOGI("Context size: %d", contextSize);
        
        // 调用 Cactus FFI 初始化
        cactus_model_t model = cactus_init(
            model_path.c_str(),
            static_cast<size_t>(contextSize),
            nullptr  // corpus_dir
        );
        
        if (!model) {
            LOGE("Failed to initialize model");
            return 0;
        }
        
        // 创建模型句柄
        auto handle = std::make_unique<ModelHandle>();
        handle->model = model;
        handle->model_path = model_path;
        handle->context_size = contextSize;
        
        // 分配模型 ID
        int64_t model_id;
        {
            std::lock_guard<std::mutex> lock(g_models_mutex);
            model_id = g_next_model_id++;
            g_models[model_id] = std::move(handle);
        }
        
        LOGI("Model initialized successfully, ID: %lld", model_id);
        return static_cast<jlong>(model_id);
        
    } catch (const std::exception& e) {
        LOGE("Exception in initFromGGUF: %s", e.what());
        return 0;
    }
}

/**
 * 文本补全（阻塞式）
 */
JNIEXPORT jstring JNICALL
Java_com_nndeploy_cactus_CactusEngine_nativeComplete(
    JNIEnv* env,
    jobject /* this */,
    jlong modelId,
    jstring prompt,
    jint maxTokens,
    jfloat temperature,
    jfloat topP,
    jint topK
) {
    try {
        auto* handle = get_model_handle(modelId);
        if (!handle) {
            LOGE("Invalid model ID: %lld", modelId);
            return string_to_jstring(env, "");
        }
        
        std::lock_guard<std::mutex> lock(handle->mutex);
        
        std::string prompt_str = jstring_to_string(env, prompt);
        
        // 构建 messages JSON
        std::string messages = "[{\"role\": \"user\", \"content\": \"" + 
                              prompt_str + "\"}]";
        
        // 构建 options JSON
        char options[256];
        snprintf(options, sizeof(options),
                "{\"max_tokens\": %d, \"temperature\": %.2f, "
                "\"top_p\": %.2f, \"top_k\": %d}",
                maxTokens, temperature, topP, topK);
        
        // 调用 Cactus 推理
        char response_buffer[8192];
        int result = cactus_complete(
            handle->model,
            messages.c_str(),
            response_buffer,
            sizeof(response_buffer),
            options,
            nullptr,  // tools
            nullptr,  // callback
            nullptr   // user_data
        );
        
        if (result <= 0) {
            LOGE("Inference failed with code: %d", result);
            return string_to_jstring(env, "");
        }
        
        // 解析 JSON 响应
        // 简化版：直接返回 response 字段
        // TODO: 使用 JSON 库解析
        std::string response(response_buffer);
        
        return string_to_jstring(env, response);
        
    } catch (const std::exception& e) {
        LOGE("Exception in complete: %s", e.what());
        return string_to_jstring(env, "");
    }
}

/**
 * Token 回调数据结构
 */
struct StreamingContext {
    JNIEnv* env;
    jobject callback;
    jmethodID onTokenMethod;
    jmethodID onCompleteMethod;
    jmethodID onErrorMethod;
};

/**
 * Token 回调函数
 */
void token_callback(const char* token, uint32_t token_id, void* user_data) {
    auto* ctx = static_cast<StreamingContext*>(user_data);
    JNIEnv* env = ctx->env;
    
    if (token && ctx->callback && ctx->onTokenMethod) {
        jstring jtoken = env->NewStringUTF(token);
        env->CallVoidMethod(
            ctx->callback,
            ctx->onTokenMethod,
            jtoken,
            static_cast<jint>(token_id)
        );
        env->DeleteLocalRef(jtoken);
    }
}

/**
 * 流式文本补全
 */
JNIEXPORT jint JNICALL
Java_com_nndeploy_cactus_CactusEngine_nativeCompleteStreaming(
    JNIEnv* env,
    jobject /* this */,
    jlong modelId,
    jstring prompt,
    jobject callback
) {
    try {
        auto* handle = get_model_handle(modelId);
        if (!handle) {
            LOGE("Invalid model ID: %lld", modelId);
            return -1;
        }
        
        std::lock_guard<std::mutex> lock(handle->mutex);
        
        std::string prompt_str = jstring_to_string(env, prompt);
        std::string messages = "[{\"role\": \"user\", \"content\": \"" + 
                              prompt_str + "\"}]";
        
        // 获取回调方法
        jclass callback_class = env->GetObjectClass(callback);
        StreamingContext ctx;
        ctx.env = env;
        ctx.callback = callback;
        ctx.onTokenMethod = env->GetMethodID(
            callback_class, "onToken", "(Ljava/lang/String;I)V"
        );
        ctx.onCompleteMethod = env->GetMethodID(
            callback_class, "onComplete", "(Ljava/lang/String;)V"
        );
        ctx.onErrorMethod = env->GetMethodID(
            callback_class, "onError", "(Ljava/lang/String;)V"
        );
        
        // 执行流式推理
        char response_buffer[8192];
        int result = cactus_complete(
            handle->model,
            messages.c_str(),
            response_buffer,
            sizeof(response_buffer),
            nullptr,
            nullptr,
            token_callback,
            &ctx
        );
        
        // 调用完成回调
        if (result > 0 && ctx.onCompleteMethod) {
            jstring jresponse = env->NewStringUTF(response_buffer);
            env->CallVoidMethod(ctx.callback, ctx.onCompleteMethod, jresponse);
            env->DeleteLocalRef(jresponse);
        } else if (ctx.onErrorMethod) {
            const char* error = cactus_get_last_error();
            jstring jerror = env->NewStringUTF(error ? error : "Unknown error");
            env->CallVoidMethod(ctx.callback, ctx.onErrorMethod, jerror);
            env->DeleteLocalRef(jerror);
        }
        
        return result > 0 ? 0 : -1;
        
    } catch (const std::exception& e) {
        LOGE("Exception in completeStreaming: %s", e.what());
        return -1;
    }
}

/**
 * 获取模型信息
 */
JNIEXPORT jstring JNICALL
Java_com_nndeploy_cactus_CactusEngine_nativeGetModelInfo(
    JNIEnv* env,
    jobject /* this */,
    jlong modelId
) {
    try {
        auto* handle = get_model_handle(modelId);
        if (!handle) {
            return string_to_jstring(env, "{}");
        }
        
        // 构建 JSON 信息
        char info[512];
        snprintf(info, sizeof(info),
                "{\"model_path\": \"%s\", \"context_size\": %zu}",
                handle->model_path.c_str(),
                handle->context_size);
        
        return string_to_jstring(env, info);
        
    } catch (const std::exception& e) {
        LOGE("Exception in getModelInfo: %s", e.what());
        return string_to_jstring(env, "{}");
    }
}

/**
 * 重置模型状态
 */
JNIEXPORT void JNICALL
Java_com_nndeploy_cactus_CactusEngine_nativeReset(
    JNIEnv* /* env */,
    jobject /* this */,
    jlong modelId
) {
    try {
        auto* handle = get_model_handle(modelId);
        if (handle) {
            std::lock_guard<std::mutex> lock(handle->mutex);
            cactus_reset(handle->model);
            LOGI("Model %lld reset", modelId);
        }
    } catch (const std::exception& e) {
        LOGE("Exception in reset: %s", e.what());
    }
}

/**
 * 释放模型
 */
JNIEXPORT void JNICALL
Java_com_nndeploy_cactus_CactusEngine_nativeDestroy(
    JNIEnv* /* env */,
    jobject /* this */,
    jlong modelId
) {
    try {
        std::lock_guard<std::mutex> lock(g_models_mutex);
        auto it = g_models.find(modelId);
        if (it != g_models.end()) {
            cactus_destroy(it->second->model);
            g_models.erase(it);
            LOGI("Model %lld destroyed", modelId);
        }
    } catch (const std::exception& e) {
        LOGE("Exception in destroy: %s", e.what());
    }
}

} // extern "C"
```

#### Java 接口封装

创建 `ffi/java/com/nndeploy/cactus/CactusEngine.java`：

```java
package com.nndeploy.cactus;

import android.util.Log;

/**
 * Cactus 推理引擎 Java 接口
 */
public class CactusEngine {
    private static final String TAG = "CactusEngine";
    
    static {
        try {
            System.loadLibrary("c++_shared");
            System.loadLibrary("cactus_engine");
            System.loadLibrary("nndeploy_framework");
            Log.i(TAG, "Native libraries loaded successfully");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "Failed to load native libraries", e);
            throw e;
        }
    }
    
    private long mModelId = 0;
    
    /**
     * 从 GGUF 文件初始化模型
     * 
     * @param ggufPath GGUF 模型文件路径
     * @param contextSize 上下文窗口大小（建议 2048 或 4096）
     * @return true 成功, false 失败
     */
    public boolean initFromGGUF(String ggufPath, int contextSize) {
        mModelId = nativeInitFromGGUF(ggufPath, contextSize);
        return mModelId > 0;
    }
    
    /**
     * 文本补全（阻塞式）
     */
    public String complete(String prompt, int maxTokens, 
                          float temperature, float topP, int topK) {
        if (mModelId <= 0) {
            throw new IllegalStateException("Model not initialized");
        }
        return nativeComplete(mModelId, prompt, maxTokens, 
                            temperature, topP, topK);
    }
    
    /**
     * 文本补全（默认参数）
     */
    public String complete(String prompt) {
        return complete(prompt, 512, 0.7f, 0.9f, 40);
    }
    
    /**
     * 流式文本补全
     */
    public void completeStreaming(String prompt, StreamingCallback callback) {
        if (mModelId <= 0) {
            throw new IllegalStateException("Model not initialized");
        }
        nativeCompleteStreaming(mModelId, prompt, callback);
    }
    
    /**
     * 获取模型信息
     */
    public String getModelInfo() {
        if (mModelId <= 0) {
            return "{}";
        }
        return nativeGetModelInfo(mModelId);
    }
    
    /**
     * 重置模型状态（清除 KV Cache）
     */
    public void reset() {
        if (mModelId > 0) {
            nativeReset(mModelId);
        }
    }
    
    /**
     * 释放模型资源
     */
    public void destroy() {
        if (mModelId > 0) {
            nativeDestroy(mModelId);
            mModelId = 0;
        }
    }
    
    @Override
    protected void finalize() throws Throwable {
        try {
            destroy();
        } finally {
            super.finalize();
        }
    }
    
    // ============================================
    // Native 方法声明
    // ============================================
    
    private native long nativeInitFromGGUF(String ggufPath, int contextSize);
    private native String nativeComplete(long modelId, String prompt, 
                                        int maxTokens, float temperature, 
                                        float topP, int topK);
    private native int nativeCompleteStreaming(long modelId, String prompt,
                                              StreamingCallback callback);
    private native String nativeGetModelInfo(long modelId);
    private native void nativeReset(long modelId);
    private native void nativeDestroy(long modelId);
    
    /**
     * 流式推理回调接口
     */
    public interface StreamingCallback {
        /**
         * 每生成一个 token 时调用
         * @param token 生成的文本片段
         * @param tokenId Token ID
         */
        void onToken(String token, int tokenId);
        
        /**
         * 生成完成时调用
         * @param fullText 完整生成的文本
         */
        void onComplete(String fullText);
        
        /**
         * 发生错误时调用
         * @param error 错误信息
         */
        void onError(String error);
    }
}
```

---

## Part 2: nndeploy 后端适配

### 2.1 Backend Adapter 完整实现

创建 `framework/source/nndeploy/inference/cactus/cactus_inference.h`：

```cpp
#ifndef _NNDEPLOY_INFERENCE_CACTUS_CACTUS_INFERENCE_H_
#define _NNDEPLOY_INFERENCE_CACTUS_CACTUS_INFERENCE_H_

#include "nndeploy/inference/inference.h"
#include "nndeploy/base/status.h"
#include "cactus/ffi/cactus_ffi.h"

namespace nndeploy {
namespace inference {

/**
 * Cactus 推理后端实现
 * 
 * 支持 GGUF 格式模型的加载和推理
 */
class NNDEPLOY_CC_API CactusInference : public Inference {
public:
    CactusInference(base::InferenceType type);
    virtual ~CactusInference();

    // ========================================
    // Inference 接口实现
    // ========================================
    
    virtual base::Status init() override;
    virtual base::Status deinit() override;
    
    virtual base::Status reshape(base::ShapeMap& shape_map) override;
    virtual base::Status run() override;
    
    virtual device::Tensor* getInputTensor(const std::string& name) override;
    virtual device::Tensor* getOutputTensor(const std::string& name) override;
    
    virtual base::Status setInputTensor(const std::string& name,
                                       device::Tensor* tensor) override;
    
    // ========================================
    // Cactus 特定接口
    // ========================================
    
    /**
     * 设置模型路径（GGUF 文件）
     */
    base::Status setModelPath(const std::string& path);
    
    /**
     * 设置上下文窗口大小
     */
    base::Status setContextSize(size_t size);
    
    /**
     * 设置生成参数
     */
    base::Status setGenerationParams(int max_tokens,
                                     float temperature = 0.7f,
                                     float top_p = 0.9f,
                                     int top_k = 40);
    
    /**
     * 获取模型信息
     */
    std::string getModelInfo() const;
    
    /**
     * 重置模型状态
     */
    base::Status reset();

private:
    // Cactus 模型句柄
    cactus_model_t cactus_model_;
    
    // 模型配置
    std::string model_path_;
    size_t context_size_;
    
    // 生成参数
    int max_tokens_;
    float temperature_;
    float top_p_;
    int top_k_;
    
    // 输入输出
    std::map<std::string, device::Tensor*> input_tensors_;
    std::map<std::string, device::Tensor*> output_tensors_;
    
    // 状态标志
    bool is_initialized_;
    
    // 辅助函数
    base::Status loadModel();
    base::Status prepareInputMessages(std::string& messages);
    base::Status parseOutputResponse(const char* response);
};

} // namespace inference
} // namespace nndeploy

#endif // _NNDEPLOY_INFERENCE_CACTUS_CACTUS_INFERENCE_H_
```

实现文件将在下一部分继续...

---

**当前进度**: 已完成 Part 1 (Cactus 源码集成详解) 和 Part 2.1 (Backend Adapter 头文件)

**下一部分将包含**:
- 2.1 Backend Adapter 完整实现（.cc 文件）
- 2.2 GGUF 模型加载器详解
- 2.3 内存管理和优化

需要我继续生成下一部分吗？
