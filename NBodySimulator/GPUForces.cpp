#include "GPUForces.h"
#include <iostream>
#include <cmath>

// ─── OpenCL Kernel Source ─────────────────────────────────────────────────────
// This runs on the GPU — one thread per particle i
// Each thread computes the total force on particle i from all other particles
static const char* kernelSource = R"CL(
__kernel void computeForces(
    __global const float* posX,
    __global const float* posY,
    __global const float* masses,
    __global float*       forceX,
    __global float*       forceY,
    const int             n,
    const float           G,
    const float           epsilon)
{
    int i = get_global_id(0);  // each GPU thread handles one particle
    if (i >= n) return;

    float fx = 0.0f;
    float fy = 0.0f;

    float xi = posX[i];
    float yi = posY[i];
    float mi = masses[i];

    for (int j = 0; j < n; j++)
    {
        if (i == j) continue;

        float dx     = posX[j] - xi;
        float dy     = posY[j] - yi;
        float distSq = dx*dx + dy*dy + epsilon*epsilon;
        float dist   = sqrt(distSq);
        float fMag   = G * mi * masses[j] / distSq;

        fx += (fMag / dist) * dx;
        fy += (fMag / dist) * dy;
    }

    forceX[i] = fx;
    forceY[i] = fy;
}
)CL";

// ─── Constructor / Destructor ─────────────────────────────────────────────────

GPUForces::GPUForces() {}

GPUForces::~GPUForces()
{
    releaseBuffers();
    if (kernel)  clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (queue)   clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
}

// ─── Init ─────────────────────────────────────────────────────────────────────

bool GPUForces::init()
{
    cl_int err;

    // Find Intel GPU platform specifically
    cl_uint numPlatforms = 0;
    clGetPlatformIDs(0, nullptr, &numPlatforms);

    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    // Pick the platform that has a GPU device
    for (auto& p : platforms)
    {
        cl_uint numDevices = 0;
        clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
        if (numDevices > 0)
        {
            platform = p;
            clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
            break;
        }
    }

    if (!device)
    {
        std::cerr << "[GPU] No GPU device found, falling back to CPU\n";
        // Fall back to CPU OpenCL device
        clGetPlatformIDs(1, &platform, nullptr);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
    }

    // Print device name
    char name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    std::cout << "[GPU] Using device: " << name << "\n";

    // Create context and command queue
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) { std::cerr << "[GPU] Context failed\n"; return false; }

    queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    if (err != CL_SUCCESS) { std::cerr << "[GPU] Queue failed\n"; return false; }

    // Compile kernel
    program = clCreateProgramWithSource(context, 1, &kernelSource,
        nullptr, &err);
    if (err != CL_SUCCESS) { std::cerr << "[GPU] Program failed\n"; return false; }

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        // Print build log so you can see what went wrong
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
            0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
            logSize, log.data(), nullptr);
        std::cerr << "[GPU] Build error:\n" << log.data() << "\n";
        return false;
    }

    kernel = clCreateKernel(program, "computeForces", &err);
    if (err != CL_SUCCESS) { std::cerr << "[GPU] Kernel failed\n"; return false; }

    initialized = true;
    std::cout << "[GPU] OpenCL initialized successfully\n";
    return true;
}

// ─── Buffer Management ────────────────────────────────────────────────────────

void GPUForces::releaseBuffers()
{
    if (bufPosX) { clReleaseMemObject(bufPosX);   bufPosX = nullptr; }
    if (bufPosY) { clReleaseMemObject(bufPosY);   bufPosY = nullptr; }
    if (bufMasses) { clReleaseMemObject(bufMasses); bufMasses = nullptr; }
    if (bufForceX) { clReleaseMemObject(bufForceX); bufForceX = nullptr; }
    if (bufForceY) { clReleaseMemObject(bufForceY); bufForceY = nullptr; }
    allocatedSize = 0;
}

void GPUForces::allocateBuffers(int n)
{
    releaseBuffers();
    bufPosX = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), nullptr, nullptr);
    bufPosY = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), nullptr, nullptr);
    bufMasses = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), nullptr, nullptr);
    bufForceX = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr, nullptr);
    bufForceY = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr, nullptr);
    allocatedSize = n;
}

// ─── Main Compute Function ────────────────────────────────────────────────────

void GPUForces::compute(std::vector<Particle>& particles, float G, float epsilon)
{
    if (!initialized) return;

    const int n = static_cast<int>(particles.size());
    if (n == 0) return;

    // Reallocate buffers if particle count changed (due to merging)
    if (n != allocatedSize)
        allocateBuffers(n);

    // Pack particle data into flat arrays for GPU transfer
    std::vector<float> posX(n), posY(n), masses(n);
    for (int i = 0; i < n; i++)
    {
        posX[i] = particles[i].position.x;
        posY[i] = particles[i].position.y;
        masses[i] = particles[i].mass;
    }

    // Upload CPU → GPU
    clEnqueueWriteBuffer(queue, bufPosX, CL_TRUE, 0,
        n * sizeof(float), posX.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, bufPosY, CL_TRUE, 0,
        n * sizeof(float), posY.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, bufMasses, CL_TRUE, 0,
        n * sizeof(float), masses.data(), 0, nullptr, nullptr);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufPosX);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufPosY);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufMasses);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufForceX);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &bufForceY);
    clSetKernelArg(kernel, 5, sizeof(int), &n);
    clSetKernelArg(kernel, 6, sizeof(float), &G);
    clSetKernelArg(kernel, 7, sizeof(float), &epsilon);

    // Launch kernel — one GPU thread per particle
    size_t globalSize = static_cast<size_t>(n);
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
        &globalSize, nullptr, 0, nullptr, nullptr);

    // Wait for GPU to finish
    clFinish(queue);

    // Download GPU → CPU
    std::vector<float> forceX(n), forceY(n);
    clEnqueueReadBuffer(queue, bufForceX, CL_TRUE, 0,
        n * sizeof(float), forceX.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, bufForceY, CL_TRUE, 0,
        n * sizeof(float), forceY.data(), 0, nullptr, nullptr);

    // Write forces back into particles
    for (int i = 0; i < n; i++)
    {
        particles[i].force.x += forceX[i];
        particles[i].force.y += forceY[i];
    }
}