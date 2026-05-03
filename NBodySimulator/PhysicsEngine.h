#pragma once
#include <vector>
#include "Particle.h"
#include "QuadTree.h"
#include "GPUForces.h"

class PhysicsEngine
{
public:
    std::vector<Particle> particles;
    bool useBarnesHut = false;
    bool useMPI = false;
    bool useGPU = false;    // Phase 6: new

    void update(float dt);
    void computeMPIForces(int rank, int numProcs);
    void integrate_public(float dt) { integrate(dt); }
    void handleMerging_public() { handleMerging(); }

    // Call once at startup
    bool initGPU() { return gpu.init(); }
    bool gpuReady() const { return gpu.isReady(); }

private:
    void integrate(float dt);
    void computeGravitationalForces();
    void computeBarnesHutForces();
    void handleMerging();

    QuadTree  quadTree;
    GPUForces gpu;                  // Phase 6: new

    static constexpr float G = 1.0f;
    static constexpr float EPSILON = 15.f;
};