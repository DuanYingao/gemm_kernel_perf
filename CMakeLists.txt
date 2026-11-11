cmake_minimum_required(VERSION 3.21)

#声明工程名称
project(gemm_kernel LANGUAGES CXX CUDA)

#bin 目录下存放exe文件
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

#把src目录下的cpp文件全部合到SRC_LIST中
aux_source_directory(src SRC_LIST)#SRC_LIST的名称自定

#指定头文件路径
include_directories(include)

#cublas 库依赖
find_package(CUDAToolkit REQUIRED)


#声明要生成的可执行文件和文件的依赖
add_executable(wmma_share ./wmma/wmma_share.cu)#不用写.h

#target_link_libraries(cublas PRIVATE CUDA::cublas)
#或者
#add_executable(main ${SRC_LIST})#不用写.h