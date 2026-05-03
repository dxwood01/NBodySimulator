#include "Benchmark.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <fstream>

static void setupParticles(PhysicsEngine& engine, int n, int rank)
{
    engine.particles.clear();

    if (rank == 0)
    {
        for (int i = 0; i < n; i++)
        {
            float x = static_cast<float>(rand() % 1000);
            float y = static_cast<float>(rand() % 800);
            engine.particles.emplace_back(
                sf::Vector2f{ x, y },
                sf::Vector2f{ 0.f, 0.f },
                2.f,
                10.f + static_cast<float>(rand() % 40)
            );
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
        engine.particles.resize(n, Particle({ 0,0 }, { 0,0 }, 1.f, 1.f));

    std::vector<float> packed(n * 4);
    if (rank == 0)
        for (int i = 0; i < n; i++)
        {
            packed[i] = engine.particles[i].position.x;
            packed[i + n] = engine.particles[i].position.y;
            packed[i + 2 * n] = engine.particles[i].mass;
            packed[i + 3 * n] = engine.particles[i].radius;
        }

    MPI_Bcast(packed.data(), n * 4, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank != 0)
        for (int i = 0; i < n; i++)
        {
            engine.particles[i].position = { packed[i], packed[i + n] };
            engine.particles[i].mass = packed[i + 2 * n];
            engine.particles[i].radius = packed[i + 3 * n];
        }
}

// All ranks call this — rank 0 does the work, others just hit barriers
static float runTest(PhysicsEngine& engine, int steps, int rank, int numProcs)
{
    const float dt = 0.016f;

    // Warmup
    if (engine.useMPI)
    {
        engine.computeMPIForces(rank, numProcs);
        if (rank == 0) { engine.integrate_public(dt); engine.handleMerging_public(); }

        int n = static_cast<int>(engine.particles.size());
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) engine.particles.resize(n, Particle({ 0,0 }, { 0,0 }, 1.f, 1.f));
        std::vector<float> packed(n * 4);
        if (rank == 0)
            for (int j = 0; j < n; j++) {
                packed[j] = engine.particles[j].position.x;
                packed[j + n] = engine.particles[j].position.y;
                packed[j + 2 * n] = engine.particles[j].mass;
                packed[j + 3 * n] = engine.particles[j].radius;
            }
        MPI_Bcast(packed.data(), n * 4, MPI_FLOAT, 0, MPI_COMM_WORLD);
        if (rank != 0)
            for (int j = 0; j < n; j++) {
                engine.particles[j].position = { packed[j], packed[j + n] };
                engine.particles[j].mass = packed[j + 2 * n];
                engine.particles[j].radius = packed[j + 3 * n];
            }
    }
    else
    {
        // Serial / OpenMP / Barnes-Hut / GPU — only rank 0 works
        if (rank == 0) engine.update(dt);
    }

    // All ranks sync before timing starts
    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < steps; i++)
    {
        if (engine.useMPI)
        {
            engine.computeMPIForces(rank, numProcs);
            if (rank == 0)
            {
                engine.integrate_public(dt);
                engine.handleMerging_public();
            }

            int n = static_cast<int>(engine.particles.size());
            MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (rank != 0)
                engine.particles.resize(n, Particle({ 0,0 }, { 0,0 }, 1.f, 1.f));

            std::vector<float> packed(n * 4);
            if (rank == 0)
                for (int j = 0; j < n; j++) {
                    packed[j] = engine.particles[j].position.x;
                    packed[j + n] = engine.particles[j].position.y;
                    packed[j + 2 * n] = engine.particles[j].mass;
                    packed[j + 3 * n] = engine.particles[j].radius;
                }
            MPI_Bcast(packed.data(), n * 4, MPI_FLOAT, 0, MPI_COMM_WORLD);
            if (rank != 0)
                for (int j = 0; j < n; j++) {
                    engine.particles[j].position = { packed[j], packed[j + n] };
                    engine.particles[j].mass = packed[j + 2 * n];
                    engine.particles[j].radius = packed[j + 3 * n];
                }
        }
        else
        {
            // Serial / OpenMP / Barnes-Hut / GPU
            if (rank == 0) engine.update(dt);
        }

        // Every rank hits this barrier every step — keeps all ranks in lockstep
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(t1 - t0).count() / steps;
}

std::vector<BenchmarkResult> Benchmark::run(int rank, int numProcs,
    const std::string& mode)
{
    std::vector<BenchmarkResult> results;
    srand(static_cast<unsigned int>(time(nullptr)) + rank);

    std::vector<int> sizes = { 100, 500, 1000, 2500, 5000, 10000 };

    auto getSteps = [](int n) -> int {
        if (n <= 100)  return 50;
        if (n <= 500)  return 20;
        if (n <= 1000) return 10;
        if (n <= 2500) return 5;
        if (n <= 5000) return 3;
        return 2;
        };

    if (rank == 0)
    {
        std::cout << "\n===== BENCHMARK RESULTS (" << mode << " mode) =====\n";
        std::cout << std::setw(8) << "N"
            << std::setw(15) << "Mode"
            << std::setw(12) << "ms/frame"
            << std::setw(10) << "FPS"
            << std::setw(12) << "Speedup\n";
        std::cout << std::string(57, '-') << "\n";
    }

    for (int n : sizes)
    {
        float serialBase = 0.f;
        float bhBase = 0.f;
        int   steps = getSteps(n);

        // ── SERIAL ───────────────────────────────────────────────────────────
        MPI_Barrier(MPI_COMM_WORLD);
        {
            PhysicsEngine engine;
            setupParticles(engine, n, rank);
            engine.useGPU = false;
            engine.useMPI = false;
            engine.useBarnesHut = false;
            omp_set_num_threads(1);

            float t = runTest(engine, steps, rank, numProcs);
            serialBase = t;
            omp_set_num_threads(omp_get_max_threads());

            if (rank == 0)
            {
                results.push_back({ n, "Serial", t, 1000.f / t, 1.0f });
                std::cout << std::setw(8) << n
                    << std::setw(15) << "Serial"
                    << std::setw(12) << std::fixed << std::setprecision(2) << t
                    << std::setw(10) << std::fixed << std::setprecision(1) << 1000.f / t
                    << std::setw(12) << std::fixed << std::setprecision(3) << 1.0f
                    << "\n";
                std::cout.flush();
            }
        }

        // ── OPENMP ───────────────────────────────────────────────────────────
        MPI_Barrier(MPI_COMM_WORLD);
        {
            PhysicsEngine engine;
            setupParticles(engine, n, rank);
            engine.useGPU = false;
            engine.useMPI = false;
            engine.useBarnesHut = false;
            omp_set_num_threads(omp_get_max_threads());

            float t = runTest(engine, steps, rank, numProcs);
            float sp = serialBase > 0 ? serialBase / t : 1.f;

            if (rank == 0)
            {
                results.push_back({ n, "OpenMP", t, 1000.f / t, sp });
                std::cout << std::setw(8) << n
                    << std::setw(15) << "OpenMP"
                    << std::setw(12) << t
                    << std::setw(10) << 1000.f / t
                    << std::setw(12) << sp << "\n";
                std::cout.flush();
            }
        }

        // ── BARNES-HUT ───────────────────────────────────────────────────────
        MPI_Barrier(MPI_COMM_WORLD);
        {
            PhysicsEngine engine;
            setupParticles(engine, n, rank);
            engine.useBarnesHut = true;
            engine.useMPI = false;
            engine.useGPU = false;

            float t = runTest(engine, steps, rank, numProcs);
            bhBase = t;

            float baseline = serialBase > 0 ? serialBase : t;
            float sp = baseline / t;

            if (rank == 0)
            {
                results.push_back({ n, "Barnes-Hut", t, 1000.f / t, sp });
                std::cout << std::setw(8) << n
                    << std::setw(15) << "Barnes-Hut"
                    << std::setw(12) << t
                    << std::setw(10) << 1000.f / t
                    << std::setw(12) << sp << "\n";
                std::cout.flush();
            }
        }

        // ── MPI — only when mode == "mpi" ────────────────────────────────────
        if (mode == "mpi")
        {
            MPI_Barrier(MPI_COMM_WORLD);
            PhysicsEngine engine;
            setupParticles(engine, n, rank);
            engine.useMPI = true;
            engine.useBarnesHut = false;
            engine.useGPU = false;

            float t = runTest(engine, steps, rank, numProcs);
            float baseline = serialBase > 0 ? serialBase : bhBase;
            float sp = baseline > 0 ? baseline / t : 1.f;

            if (rank == 0)
            {
                results.push_back({ n, "MPI", t, 1000.f / t, sp });
                std::cout << std::setw(8) << n
                    << std::setw(15) << "MPI"
                    << std::setw(12) << t
                    << std::setw(10) << 1000.f / t
                    << std::setw(12) << sp << "\n";
                std::cout.flush();
            }
        }

        // ── GPU — only when mode == "gpu" ────────────────────────────────────
        if (mode == "gpu")
        {
            MPI_Barrier(MPI_COMM_WORLD);
            PhysicsEngine engine;
            if (rank == 0) engine.initGPU();
            setupParticles(engine, n, rank);
            engine.useGPU = true;
            engine.useMPI = false;
            engine.useBarnesHut = false;

            float t = 0.f;
            if (rank == 0) t = runTest(engine, steps, rank, numProcs);
            MPI_Bcast(&t, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

            float baseline = serialBase > 0 ? serialBase : bhBase;
            float sp = baseline > 0 ? baseline / t : 1.f;

            if (rank == 0)
            {
                results.push_back({ n, "GPU", t, 1000.f / t, sp });
                std::cout << std::setw(8) << n
                    << std::setw(15) << "GPU"
                    << std::setw(12) << t
                    << std::setw(10) << 1000.f / t
                    << std::setw(12) << sp << "\n";
                std::cout.flush();
            }
        }

        if (rank == 0)
        {
            std::cout << std::string(57, '-') << "\n";
            std::cout.flush();
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    return results;
}

void Benchmark::exportCSV(const std::vector<BenchmarkResult>& results,
    const std::string& filename,
    int numProcs)
{
    std::ofstream f(filename);
    if (!f.is_open())
    {
        std::cerr << "Could not open " << filename << " for writing\n";
        return;
    }

    f << "particle_count,mode,avg_ms,fps,speedup,mpi_procs\n";

    for (const auto& r : results)
    {
        f << r.particleCount << ","
            << r.mode << ","
            << std::fixed << std::setprecision(4) << r.avgMs << ","
            << std::fixed << std::setprecision(4) << r.fps << ","
            << std::fixed << std::setprecision(4) << r.speedup << ","
            << numProcs << "\n";
    }

    f.close();
    std::cout << "Results exported to: " << filename << "\n";
}