#pragma once
#include <CL/cl.h>
#include <vector>
#include "Particle.h"

class GPUForces
{
public:
    GPUForces();
    ~GPUForces();

    // Initialize OpenCL — call once at startup
    bool init();

    // Compute gravitational forces on GPU for all particles
    // Results are written directly into particle.force
    void compute(std::vector<Particle>& particles, float G, float epsilon);

    bool isReady() const { return initialized; }

private:
    bool initialized = false;

    cl_platform_id   platform = nullptr;
    cl_device_id     device = nullptr;
    cl_context       context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program       program = nullptr;
    cl_kernel        kernel = nullptr;

    // GPU buffers — allocated once, reused every frame
    cl_mem bufPosX = nullptr;
    cl_mem bufPosY = nullptr;
    cl_mem bufMasses = nullptr;
    cl_mem bufForceX = nullptr;
    cl_mem bufForceY = nullptr;

    int allocatedSize = 0;  // track buffer size to reallocate if particle count changes

    void releaseBuffers();
    void allocateBuffers(int n);
};