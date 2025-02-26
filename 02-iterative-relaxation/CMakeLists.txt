cmake_minimum_required(VERSION 3.12)
project(mpna-iter-1 CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(iter_cg main_cg.cpp
        GMRES.h
        CSRMatrix.h
        Jacobi.h
        GaussSeidel.h)
