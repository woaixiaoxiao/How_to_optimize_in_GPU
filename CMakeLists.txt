cmake_minimum_required(VERSION 3.17.0)
project(testdebug VERSION 0.1.0 LANGUAGES CUDA CXX C)

find_package(CUDAToolkit)
set(CMAKE_CUDA_ARCHITECTURES 86) 
set(CMAKE_CUDA_STANDARD 17)
# nvcc -O3 main.cu -o main -arch=compute_86 -code=sm_86 -lcublas
# 设置默认的 CUDA 架构
# set(CMAKE_CUDA_ARCHITECTURES 86)  # 这里选择 70 代表 Volta 架构，可以根据需要更改

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

file(GLOB CU_FILES "*.cu")
foreach(CU_FILE ${CU_FILES})
    # 获取文件名（不带扩展名）
    get_filename_component(FILE_NAME ${CU_FILE} NAME_WE)
    
    # 创建对应的可执行文件
    add_executable(${FILE_NAME} ${CU_FILE})
    
    # 链接 CUDA CUBLAS 库
    target_link_libraries(${FILE_NAME} PRIVATE CUDA::cudart -lcublas -lcurand)

    target_compile_options(${FILE_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=compute_86 -code=sm_86 --extended-lambda>)
    
    # 设置 CUDA 可分离编译属性
    set_target_properties(${FILE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()