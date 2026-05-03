#pragma once
#include <vector>
#include <omp.h>

namespace Parallel
{
    // Runs any callable on each element of a vector in parallel
    template<typename T, typename Func>
    void forEach(std::vector<T>& vec, Func&& func)
    {
#pragma omp parallel for
        for (int i = 0; i < (int)vec.size(); i++)
        {
            func(vec[i], i);
        }
    }
}