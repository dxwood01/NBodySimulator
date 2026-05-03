#pragma once
#include "PhysicsEngine.h"
#include <vector>
#include <string>

struct BenchmarkResult
{
    int         particleCount;
    std::string mode;
    float       avgMs;
    float       fps;
    float       speedup;
};

class Benchmark
{
public:
    static std::vector<BenchmarkResult> run(int rank, int numProcs,
        const std::string& mode);
    static void exportCSV(const std::vector<BenchmarkResult>& results,
        const std::string& filename,
        int numProcs = 1);   // ← add parameter
}; 