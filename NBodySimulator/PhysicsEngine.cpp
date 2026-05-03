#include "PhysicsEngine.h"
#include <cmath>
#include <omp.h>
#include <mpi.h>
#include <vector>
#include <chrono>
#include <iostream>


void PhysicsEngine::update(float dt)
{
    if (useGPU)
        gpu.compute(particles, G, EPSILON);   // GPU handles forces
    else if (useBarnesHut)
        computeBarnesHutForces();
    else
        computeGravitationalForces();

    integrate(dt);
    handleMerging();
}

// ─── MPI Force Distribution ───────────────────────────────────────────────────

void PhysicsEngine::computeMPIForces(int rank, int numProcs)
{

    // Step 1: Broadcast current particle count to all ranks FIRST
    // This is the critical fix — all ranks must agree on n before
    // allocating any buffers, especially after merging reduces particle count
    int n = static_cast<int>(particles.size());
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resize non-rank-0 particles to match
    if (rank != 0)
        particles.resize(n, Particle(
            sf::Vector2f{ 0,0 }, sf::Vector2f{ 0,0 }, 1.f, 1.f));

    if (n == 0) return;

    // Step 2: Pack positions and masses from rank 0
    std::vector<float> posX(n), posY(n), masses(n);

    if (rank == 0)
    {
        for (int i = 0; i < n; i++)
        {
            posX[i] = particles[i].position.x;
            posY[i] = particles[i].position.y;
            masses[i] = particles[i].mass;
        }
    }

    // Step 3: Broadcast positions and masses to all ranks
    std::vector<float> packedIn(n * 3);
    if (rank == 0)
        for (int i = 0; i < n; i++) {
            packedIn[i] = particles[i].position.x;
            packedIn[i + n] = particles[i].position.y;
            packedIn[i + 2 * n] = particles[i].mass;
        }
    MPI_Bcast(packedIn.data(), n * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (rank != 0)
        for (int i = 0; i < n; i++) {
            posX[i] = packedIn[i];
            posY[i] = packedIn[i + n];
            masses[i] = packedIn[i + 2 * n];
        }

    // Step 4: Each rank computes forces for its slice
    int sliceSize = n / numProcs;
    int startIdx = rank * sliceSize;
    int endIdx = (rank == numProcs - 1) ? n : startIdx + sliceSize;

    // ── DEBUG: verify each rank gets a different slice ──
    //std::cout << "Rank " << rank << " handling particles "
    //    << startIdx << " to " << endIdx
    //    << " out of " << n << "\n";
    // ────────────────────────────────────────────────────


    std::vector<float> localFx(n, 0.f), localFy(n, 0.f);

    for (int i = startIdx; i < endIdx; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j) continue;

            float dx = posX[j] - posX[i];
            float dy = posY[j] - posY[i];
            float distSq = dx * dx + dy * dy + EPSILON * EPSILON;
            float dist = std::sqrt(distSq);
            float fMag = G * masses[i] * masses[j] / distSq;

            localFx[i] += (fMag / dist) * dx;
            localFy[i] += (fMag / dist) * dy;
        }
    }

    // Step 5: Allreduce with correct buffer size n
    std::vector<float> totalFx(n, 0.f), totalFy(n, 0.f);
    MPI_Allreduce(localFx.data(), totalFx.data(), n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(localFy.data(), totalFy.data(), n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    // Step 6: Apply forces back to particles
    for (int i = 0; i < n; i++)
    {
        particles[i].force.x += totalFx[i];
        particles[i].force.y += totalFy[i];
    }
}

// ─── Brute Force O(n²) — Phase 3 ─────────────────────────────────────────────

void PhysicsEngine::computeGravitationalForces()
{
    const int n = static_cast<int>(particles.size());
    const int numThreads = omp_get_max_threads();

    std::vector<std::vector<sf::Vector2f>> localForces(
        numThreads,
        std::vector<sf::Vector2f>(n, sf::Vector2f(0.f, 0.f))
    );

#pragma omp parallel
    {
        int tid = omp_get_thread_num();

#pragma omp for schedule(dynamic)
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                sf::Vector2f diff = particles[j].position - particles[i].position;
                float distSq = diff.x * diff.x + diff.y * diff.y + EPSILON * EPSILON;
                float dist = std::sqrt(distSq);
                float forceMag = G * particles[i].mass * particles[j].mass / distSq;
                sf::Vector2f forceVec = (forceMag / dist) * diff;

                localForces[tid][i] += forceVec;
                localForces[tid][j] -= forceVec;
            }
        }
    }

    for (int i = 0; i < n; i++)
    {
        sf::Vector2f total(0.f, 0.f);
        for (int t = 0; t < numThreads; t++)
            total += localForces[t][i];
        particles[i].force += total;
    }
}

// ─── Barnes-Hut O(n log n) — Phase 4 ─────────────────────────────────────────

void PhysicsEngine::computeBarnesHutForces()
{
    if (particles.empty()) return;

    float minX = particles[0].position.x, maxX = minX;
    float minY = particles[0].position.y, maxY = minY;

    for (const auto& p : particles)
    {
        if (p.position.x < minX) minX = p.position.x;
        if (p.position.x > maxX) maxX = p.position.x;
        if (p.position.y < minY) minY = p.position.y;
        if (p.position.y > maxY) maxY = p.position.y;
    }

    float cx = (minX + maxX) / 2.f;
    float cy = (minY + maxY) / 2.f;
    float halfSize = std::max(maxX - minX, maxY - minY) / 2.f + 10.f;

    AABB boundary{ cx, cy, halfSize };
    quadTree.build(particles, boundary);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(particles.size()); i++)
        particles[i].force += quadTree.computeForce(i, particles, G, EPSILON);
}

// ─── Integration ──────────────────────────────────────────────────────────────

void PhysicsEngine::integrate(float dt)
{
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(particles.size()); ++i)
        particles[i].update(dt);
}

// ─── Merging ──────────────────────────────────────────────────────────────────

void PhysicsEngine::handleMerging()
{
    std::vector<bool> alive(particles.size(), true);

    for (int i = 0; i < static_cast<int>(particles.size()); i++)
    {
        if (!alive[i]) continue;
        for (int j = i + 1; j < static_cast<int>(particles.size()); j++)
        {
            if (!alive[j]) continue;

            sf::Vector2f diff = particles[j].position - particles[i].position;
            float distSq = diff.x * diff.x + diff.y * diff.y;
            float mergedist = particles[i].radius + particles[j].radius;

            if (distSq < mergedist * mergedist)
            {
                float totalMass = particles[i].mass + particles[j].mass;
                sf::Vector2f newVel = (particles[i].velocity * particles[i].mass
                    + particles[j].velocity * particles[j].mass)
                    / totalMass;

                particles[i].mass = totalMass;
                particles[i].velocity = newVel;
                particles[i].radius = 3.f * std::cbrt(particles[i].mass / 10.f);
                if (particles[i].radius > 40.f) particles[i].radius = 40.f;
                particles[i].color = sf::Color(255, 165, 0);
                alive[j] = false;
            }
        }
    }

    std::vector<Particle> survivors;
    survivors.reserve(particles.size());
    for (int i = 0; i < static_cast<int>(particles.size()); i++)
        if (alive[i]) survivors.push_back(particles[i]);
    particles = std::move(survivors);
}