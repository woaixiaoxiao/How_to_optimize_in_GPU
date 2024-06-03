cmake_minimum_required(VERSION 3.17.0)
project(testdebug VERSION 0.1.0 LANGUAGES CUDA CXX C)

find_package(CUDAToolkit)


set(CMAKE_CUDA_STANDARD 11)

file(GLOB CU_FILES "*.cu")
foreach(CU_FILE ${CU_FILES})
    # 获取文件名（不带扩展名）
    get_filename_component(FILE_NAME ${CU_FILE} NAME_WE)
    
    # 创建对应的可执行文件
    add_executable(${FILE_NAME} ${CU_FILE})
    
    # 链接 CUDA 库
    target_link_libraries(${FILE_NAME} PRIVATE CUDA::cudart)
    
    # 如果是 Debug 模式，添加 -G 编译选项
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${FILE_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
    endif()
    
    # 设置 CUDA 可分离编译属性
    set_target_properties(${FILE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()


# add_executable(testdebug hello.cu)
# target_link_libraries(testdebug PRIVATE CUDA::cudart)
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     target_compile_options(testdebug PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
# endif()
# set_target_properties(testdebug PROPERTIES CUDA_SEPARABLE_COMPILATION ON)