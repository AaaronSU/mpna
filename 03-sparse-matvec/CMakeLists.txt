cmake_minimum_required(VERSION 3.12)
project(mpna-iter-1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)

set_source_files_properties(mmio.c PROPERTIES LANGUAGE C)
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_DIRS})

add_executable(csr_matrix_vector_multiply 
        main.cpp
        mmio.c
        mmio.h
        CSRMatrix.h)

add_executable(mpi_csr_matrix_vector_multiply 
        mpi_main.cpp
        mmio.c
        mmio.h
        CSRMatrix.h)
target_link_libraries(mpi_csr_matrix_vector_multiply ${MPI_CXX_LIBRARIES})

