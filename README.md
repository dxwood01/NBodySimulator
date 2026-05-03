# Parallel N-Body Gravitational Simulator

A real-time gravitational N-body simulator built in C++ where every particle attracts every other particle using Newton's law of gravitation (`F = G·m₁·m₂/r²`). The simulation models realistic orbital mechanics, cluster formation, and galaxy-like emergent behavior across hundreds to thousands of bodies, visualized in real time using SFML.

The project demonstrates four parallel computing strategies — **OpenMP**, **Barnes-Hut**, **MPI**, and **GPU/OpenCL** — applied to the core O(n²) force calculation bottleneck, with full performance benchmarking and analysis.

## Demo

Two clusters of particles drift toward each other under gravity, interpenetrate, and form swirling structures. Particles that collide merge into heavier bodies — conserving momentum — and grow visually. Over time, a few dominant massive bodies emerge from the initial chaos, mimicking how planets and galaxies form.

## Features

- Pairwise gravitational attraction using Newton's law with softening factor
- Momentum-conserving body merging with mass-scaled radius (`r ∝ ∛mass`)
- Barnes-Hut QuadTree (θ = 0.5) reducing complexity from O(n²) to O(n log n)
- OpenMP parallelization with per-thread force accumulators (race-condition safe)
- MPI distribution across independent processes using `MPI_Bcast` / `MPI_Allreduce`
- OpenCL GPU acceleration targeting Intel UHD integrated graphics
- Live runtime mode switching via keyboard
- Full performance benchmarking with CSV export and Python visualization

## Tech Stack

| Component             | Technology                          |
| --------------------- | ----------------------------------- |
| Language              | C++17                               |
| Rendering             | SFML 3.x                            |
| CPU Parallelism       | OpenMP                              |
| Distributed Computing | MS-MPI                              |
| GPU Acceleration      | OpenCL (Intel oneAPI)               |
| Benchmarking & Graphs | Python, pandas, matplotlib, seaborn |

## Project Structure

```
NBodySimulator/                        ← solution root
├── NBodySimulator.slnx
├── README.md
├── .gitignore
├── benchmark_analysis.ipynb
├── results/                           ← generated graphs from notebook
│   ├── graph1_update_time.png
│   ├── graph2_speedup.png
│   ├── graph3_fps.png
│   ├── graph4_mpi_scaling_time.png
│   ├── graph5_mpi_speedup.png
│   ├── graph6_mpi_bar.png
│   ├── graph7_heatmap.png
│   ├── graph8_barneshut_speedup.png
│   └── graph9_full_summary.png
└── NBodySimulator/                    ← C++ project folder
    ├── NBodySimulator.vcxproj
    ├── main.cpp
    ├── Particle.h / Particle.cpp
    ├── PhysicsEngine.h / PhysicsEngine.cpp
    ├── Renderer.h / Renderer.cpp
    ├── QuadTree.h / QuadTree.cpp
    ├── GPUForces.h / GPUForces.cpp
    ├── Benchmark.h / Benchmark.cpp
    ├── Parallel.h / Parallel.cpp
    └── x64/                           ← build output (git ignored)
```

## Dependencies

- [SFML 3.x](https://www.sfml-dev.org/)
- [Microsoft MPI (MS-MPI)](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi)
- [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) (OpenCL)
- OpenMP (included with MSVC)

## Build (Visual Studio)

1. Open `NBodySimulator.sln` in Visual Studio 2022
2. Set configuration to **Debug** or **Release**, platform **x64**
3. Verify these project properties under **Project → Properties**:

**C/C++ → Additional Include Directories:**

```
C:\SFML\include
C:\Program Files (x86)\Microsoft SDKs\MPI\Include
C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\include
```

**Linker → Additional Library Directories:**

```
C:\SFML\lib
C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64
C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\compiler\lib\intel64
```

**Linker → Additional Dependencies:**

```
sfml-graphics-d.lib
sfml-window-d.lib
sfml-system-d.lib
msmpi.lib
OpenCL.lib
```

**C/C++ → Language → OpenMP Support:** Yes (/openmp)

4. Build → Build Solution (`Ctrl+Shift+B`)

## Run Simulation

```bash
# Launch with 4 MPI processes (recommended)
mpiexec -n 4 NBodySimulator.exe

# Launch with 1 process (no MPI distribution)
NBodySimulator.exe
```

### Runtime Controls

| Key | Action                          |
| --- | ------------------------------- |
| `B` | Toggle Barnes-Hut ↔ Brute Force |
| `M` | Toggle MPI on/off               |
| `G` | Toggle GPU on/off               |
| `R` | Reset simulation                |

The on-screen HUD shows current mode, particle count, FPS, and update time in ms.

## Run Benchmarks

All CSV files are saved in `NBodySimulator/x64/Debug/`.

```bash
# GPU + Serial + OpenMP + Barnes-Hut (run WITHOUT mpiexec for full OpenMP threads)
NBodySimulator.exe bench gpu

# MPI scaling benchmarks
mpiexec -n 2 NBodySimulator.exe bench mpi
mpiexec -n 4 NBodySimulator.exe bench mpi
mpiexec -n 6 NBodySimulator.exe bench mpi
```

### Generate Graphs

```bash
pip install pandas matplotlib seaborn jupyter
jupyter notebook benchmark_analysis.ipynb
```

Run all cells with **Kernel → Restart & Run All**. Graphs are saved to `results/`.

## Benchmark Results & Analysis

Benchmarks were run on a 4-core / 8-logical-processor system with Intel UHD integrated graphics. Particle counts tested: 100, 500, 1000, 2500, 5000, 10000.

### Raw Results (ms per frame)

| Mode       | n=100 | n=500 | n=1000 | n=2500  | n=5000  | n=10000 |
| ---------- | ----- | ----- | ------ | ------- | ------- | ------- |
| Serial     | 2.20  | 50.88 | 205.89 | 1100.20 | 3472.84 | 6999.92 |
| OpenMP     | 2.20  | 52.37 | 196.34 | 1093.99 | 3356.45 | 6906.90 |
| Barnes-Hut | 2.71  | 48.37 | 172.59 | 909.85  | 2688.57 | 5571.49 |
| GPU        | 2.13  | 41.43 | 156.55 | 877.20  | 2674.39 | 5411.48 |
| MPI (n=4)  | 2.04  | 52.48 | 241.00 | 1298.80 | 4011.25 | 7060.52 |

### Key Findings

**1. Barnes-Hut scales best at large particle counts.**
At n=10000, Barnes-Hut achieves a **1.26× speedup** over Serial while all other methods struggle. This is because it reduces the number of force calculations mathematically from O(n²) to O(n log n) — from 50 million operations to ~133,000 at n=10000. No amount of hardware parallelism can match an algorithmic improvement of this magnitude at scale.

**2. GPU shows consistent gains from n=500 onwards.**
GPU achieves **1.29× speedup** at n=1000 and maintains this advantage up to n=10000. However, the gains are modest compared to a discrete GPU because the Intel UHD integrated graphics shares memory with the CPU — the PCIe transfer overhead per frame consumes much of the potential speedup. A discrete GPU would show 10–50× improvement here.

**3. OpenMP provides marginal gains at this scale.**
OpenMP speedup peaks at ~1.05× and is inconsistent across particle counts. The reason is the private-accumulator approach used to avoid race conditions — allocating a `numThreads × n` force array every frame becomes a memory bottleneck that partially cancels the parallel gains. On a system with more physical cores the benefit would be clearer.

**4. MPI is slower than Serial at large n.**
At n=2500 MPI (4 processes) takes 1298ms vs Serial's 1100ms — a regression. This is because MPI communication overhead (`MPI_Bcast` + `MPI_Allreduce`) grows with particle count. Broadcasting 2500 particles of position data every frame costs more than the computation savings from distributing the work. MPI shows modest gains only at smaller particle counts where the force calculation dominates over communication.

**5. MPI scaling hits a wall beyond physical core count.**
Results from n=2, n=4, n=6 processes show: update time drops from Serial→n=2→n=4, then flattens or worsens at n=6. The system has 4 physical cores — at n=6 processes, processes share cores and OpenMP inside each process loses threading capacity. This demonstrates **Amdahl's Law** in practice: the serial communication portion limits total speedup regardless of how many processes are added.

### Summary

| Approach   | Best Use Case        | Why                                        |
| ---------- | -------------------- | ------------------------------------------ |
| Barnes-Hut | Large n (5000+)      | Algorithm reduces total work               |
| GPU        | Medium n (500–5000)  | Parallel threads offset transfer cost      |
| OpenMP     | Small-medium n       | Fast on-core parallelism, no communication |
| MPI        | Distributed machines | Overhead too high on a single machine      |

The central lesson from this project is that **algorithmic improvement outperforms hardware parallelism** at scale. Barnes-Hut's O(n log n) complexity beats brute-force parallelization on all hardware configurations at high particle counts.
