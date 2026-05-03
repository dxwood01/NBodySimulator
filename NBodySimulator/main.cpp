#include <SFML/Graphics.hpp>
#include <SFML/System/Clock.hpp>
#include "PhysicsEngine.h"
#include "Renderer.h"
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <string>
#include <chrono>
#include <iomanip>
#include "Benchmark.h"

int main(int argc, char** argv)
{

    MPI_Init(&argc, &argv);

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    int threadsPerRank = omp_get_max_threads() / numProcs;
    if (threadsPerRank < 1) threadsPerRank = 1;
    omp_set_num_threads(threadsPerRank);

    if (argc > 1 && std::string(argv[1]) == "bench")
    {
        std::string mode = "all";
        if (argc > 2)
            mode = argv[2];

        if (rank == 0)
            std::cout << "Running benchmark mode: " << mode << "\n";

        auto results = Benchmark::run(rank, numProcs, mode);

        // Export CSV on rank 0 only
        if (rank == 0)
        {
            
            std::string csvName = "benchmark_" + mode + "_n" + std::to_string(numProcs) + ".csv";
            Benchmark::exportCSV(results, csvName);
        }

        MPI_Finalize();
        return 0;
    }

    srand(static_cast<unsigned int>(time(NULL)) + rank);

    sf::RenderWindow* window = nullptr;
    sf::Font font;
    sf::Text* statsText = nullptr;
    bool fontLoaded = false;
    Renderer renderer;

    if (rank == 0)
    {
        window = new sf::RenderWindow(sf::VideoMode({ 1200, 800 }), "N-Body Simulator");
        window->setFramerateLimit(60);

        fontLoaded = font.openFromFile("C:/Windows/Fonts/arial.ttf");
        statsText = new sf::Text(font);
        statsText->setCharacterSize(16);
        statsText->setFillColor(sf::Color::White);
        statsText->setPosition({ 10.f, 10.f });

        std::cout << "MPI processes : " << numProcs << "\n";
        std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
        std::cout << "[B] Toggle Brute Force / Barnes-Hut\n";
        std::cout << "[M] Toggle MPI on/off\n";
        std::cout << "[G] Toggle GPU on/off\n";
        std::cout << "[R] Reset simulation\n";
    }

    PhysicsEngine engine;
    engine.useMPI = true;

    // 🔥 Initialize GPU
    bool gpuOK = engine.initGPU();
    if (rank == 0)
    {
        if (gpuOK)
            std::cout << "[GPU] Ready\n";
        else
            std::cout << "[GPU] Failed\n";
    }

    // ───────────────── Simulation Setup ─────────────────
    auto setupSimulation = [&]()
        {
            engine.particles.clear();
            int n = 0;

            if (rank == 0)
            {
                auto spawnCluster = [&](float cx, float cy, float dvx, float dvy, int count)
                    {
                        for (int i = 0; i < count; i++)
                        {
                            float angle = static_cast<float>(rand()) / RAND_MAX * 2.f * 3.14159f;
                            float dist = 30.f + static_cast<float>(rand()) / RAND_MAX * 180.f;

                            float x = cx + dist * std::cos(angle);
                            float y = cy + dist * std::sin(angle);

                            float speed = static_cast<float>(rand()) / RAND_MAX * 3.f;

                            float vx = dvx + std::cos(angle) * speed;
                            float vy = dvy + std::sin(angle) * speed;

                            float radius = 2.f + static_cast<float>(rand()) / RAND_MAX * 2.f;
                            float mass = 10.f + static_cast<float>(rand()) / RAND_MAX * 40.f;

                            engine.particles.emplace_back(
                                sf::Vector2f{ x, y },
                                sf::Vector2f{ vx, vy },
                                radius, mass
                            );
                        }
                    };

                spawnCluster(300.f, 400.f, 12.f, 0.f, 500);
                spawnCluster(900.f, 400.f, -12.f, 0.f, 500);

                n = static_cast<int>(engine.particles.size());
            }

            MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

            if (rank != 0)
                engine.particles.resize(n, Particle({ 0,0 }, { 0,0 }, 1.f, 1.f));

            std::vector<float> packed(n * 6);

            if (rank == 0)
            {
                for (int i = 0; i < n; i++)
                {
                    packed[i] = engine.particles[i].position.x;
                    packed[i + n] = engine.particles[i].position.y;
                    packed[i + 2 * n] = engine.particles[i].velocity.x;
                    packed[i + 3 * n] = engine.particles[i].velocity.y;
                    packed[i + 4 * n] = engine.particles[i].radius;
                    packed[i + 5 * n] = engine.particles[i].mass;
                }
            }

            MPI_Bcast(packed.data(), n * 6, MPI_FLOAT, 0, MPI_COMM_WORLD);

            if (rank != 0)
            {
                for (int i = 0; i < n; i++)
                {
                    engine.particles[i].position = { packed[i], packed[i + n] };
                    engine.particles[i].velocity = { packed[i + 2 * n], packed[i + 3 * n] };
                    engine.particles[i].radius = packed[i + 4 * n];
                    engine.particles[i].mass = packed[i + 5 * n];
                }
            }
        };

    setupSimulation();

    // ───────────────── Frame State ─────────────────
    sf::Clock clock;
    sf::Clock fpsClock;

    int frameCount = 0;
    float fps = 0.f;
    float updateMs = 0.f;

    // flags: run, BH, MPI, reset, GPU
    int flags[5] = { 1, 1, 1, 0, 0 };

    while (flags[0])
    {
        // ── Input (rank 0)
        if (rank == 0 && window)
        {
            flags[3] = 0;

            while (auto event = window->pollEvent())
            {
                if (event->is<sf::Event::Closed>())
                    flags[0] = 0;

                if (auto* key = event->getIf<sf::Event::KeyPressed>())
                {
                    if (key->code == sf::Keyboard::Key::B)
                        flags[1] = !flags[1];

                    if (key->code == sf::Keyboard::Key::M)
                    {
                        flags[2] = !flags[2];
                        if (flags[2]) flags[4] = 0; // disable GPU
                    }

                    if (key->code == sf::Keyboard::Key::G)
                    {
                        flags[4] = !flags[4];
                        if (flags[4]) flags[2] = 0; // disable MPI
                    }

                    if (key->code == sf::Keyboard::Key::R)
                        flags[3] = 1;
                }
            }
        }

        MPI_Bcast(flags, 5, MPI_INT, 0, MPI_COMM_WORLD);

        if (!flags[0]) break;

        if (flags[3])
        {
            setupSimulation();
            clock.restart();
            continue;
        }

        engine.useBarnesHut = flags[1];
        engine.useMPI = flags[2];
        engine.useGPU = (flags[4] && engine.gpuReady());

        // safety
        if (engine.useGPU) engine.useMPI = false;
        if (engine.useMPI) engine.useGPU = false;

        auto t0 = std::chrono::high_resolution_clock::now();

        // ── Physics
        if (engine.useGPU)
        {
            if (rank == 0)
            {
                float dt = clock.restart().asSeconds();
                if (dt > 0.05f) dt = 0.05f;
                engine.update(dt);
            }
        }
        else if (engine.useMPI)
        {
            engine.computeMPIForces(rank, numProcs);

            if (rank == 0)
            {
                engine.integrate_public(0.016f);
                engine.handleMerging_public();
            }

            int n = static_cast<int>(engine.particles.size());
            MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

            if (rank != 0)
                engine.particles.resize(n, Particle({ 0,0 }, { 0,0 }, 1.f, 1.f));

            std::vector<float> packed(n * 6);

            if (rank == 0)
                for (int i = 0; i < n; i++)
                {
                    packed[i] = engine.particles[i].position.x;
                    packed[i + n] = engine.particles[i].position.y;
                    packed[i + 2 * n] = engine.particles[i].velocity.x;
                    packed[i + 3 * n] = engine.particles[i].velocity.y;
                    packed[i + 4 * n] = engine.particles[i].radius;
                    packed[i + 5 * n] = engine.particles[i].mass;
                }

            MPI_Bcast(packed.data(), n * 6, MPI_FLOAT, 0, MPI_COMM_WORLD);

            if (rank != 0)
                for (int i = 0; i < n; i++)
                {
                    engine.particles[i].position = { packed[i], packed[i + n] };
                    engine.particles[i].velocity = { packed[i + 2 * n], packed[i + 3 * n] };
                    engine.particles[i].radius = packed[i + 4 * n];
                    engine.particles[i].mass = packed[i + 5 * n];
                }
        }
        else
        {
            if (rank == 0)
            {
                float dt = clock.restart().asSeconds();
                if (dt > 0.05f) dt = 0.05f;
                engine.update(dt);
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        updateMs = std::chrono::duration<float, std::milli>(t1 - t0).count();

        // ── Console stats
        if (rank == 0)
        {
            static int counter = 0;
            static float total = 0.f;

            total += updateMs;
            counter++;

            if (counter >= 10)
            {
                std::cout << "[Frame " << std::setw(6) << frameCount << "] "
                    << "Particles: " << engine.particles.size()
                    << " | Algo: " << (engine.useBarnesHut ? "Barnes-Hut" : "Brute Force")
                    << " | MPI: " << (engine.useMPI ? "ON" : "OFF")
                    << " | GPU: " << (engine.useGPU ? "ON" : "OFF")
                    << " | Avg Update: " << std::fixed << std::setprecision(3)
                    << (total / counter) << " ms"
                    << " | FPS: " << static_cast<int>(fps)
                    << "\n";

                counter = 0;
                total = 0.f;
            }
        }

        // ── Rendering
        if (rank == 0 && window)
        {
            frameCount++;

            if (fpsClock.getElapsedTime().asSeconds() >= 0.5f)
            {
                fps = frameCount / fpsClock.restart().asSeconds();
                frameCount = 0;
            }

            if (fontLoaded && statsText)
            {
                std::string info =
                    std::string("Algorithm : ") +
                    (engine.useBarnesHut ? "Barnes-Hut O(n log n)" : "Brute Force O(n^2)") + "\n" +
                    "MPI       : " + std::string(engine.useMPI ? "ON" : "OFF") +
                    " (" + std::to_string(numProcs) + " processes)\n" +
                    "GPU       : " + std::string(engine.useGPU ? "ON" : "OFF") + "\n" +
                    "Particles : " + std::to_string(engine.particles.size()) + "\n" +
                    "FPS       : " + std::to_string((int)fps) + "\n" +
                    "Update    : " + std::to_string(updateMs).substr(0, 6) + " ms\n" +
                    "[B] Brute/Brute Force   [M] MPI   [G] GPU   [R] Reset";

                statsText->setString(info);
            }

            window->clear(sf::Color::Black);
            renderer.draw(*window, engine.particles);
            if (fontLoaded && statsText) window->draw(*statsText);
            window->display();
        }
    }

    if (rank == 0)
    {
        delete window;
        delete statsText;
    }

    MPI_Finalize();
    return 0;
}