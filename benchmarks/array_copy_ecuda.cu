#include <iostream>
#include <random>
#include <vector>
#include <chrono>

#include "../include/ecuda/event.hpp"
#include "../include/ecuda/array.hpp"
#include "../include/ecuda/matrix.hpp"

#include "config.hpp"
#ifndef BENCHMARK_THREADS
#define BENCHMARK_THREADS 480
#endif

typedef double value_type;

template<typename T,std::size_t N>
    __global__
void copyArray( typename ecuda::array<T,N>::const_kernel_argument src, typename ecuda::array<T,N>::kernel_argument dest )
{
    const int t = blockIdx.x*blockDim.x+threadIdx.x;
    if( t < src.size() ) dest[t] = src[t];
}

int main( int argc, char* argv[] )
{
    std::vector<int> dims = {50,100,200,500,1000,2000,5000,10000};
    for (int dim : dims) {
        ecuda::matrix<float> a(dim, 1), b(dim, 1);

        std::vector<float> a_cpu;
        a_cpu.resize(dim);
        static std::random_device rd;
        static std::mt19937 mt(rd());
        static std::uniform_real_distribution<> dist(-10, 10);
        for (int i =0;i<dim; ++i) {
            a_cpu[i] = dist(mt);
        }
        ecuda::copy(a_cpu.begin(), a_cpu.end(), a.begin());

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        for (int i=0; i<1000000; ++i) {
            ecuda::copy(a.begin(), a.end(), b.begin());
        }

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "dim:" << dim << " time:" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000 << std::endl;
    }

    return EXIT_SUCCESS;

}

