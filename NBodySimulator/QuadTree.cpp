#include "QuadTree.h"
#include <cmath>
#include <vector>

void QuadTree::build(const std::vector<Particle>& particles, AABB boundary)
{
    nodes.clear();
    nodes.reserve(particles.size() * 4);
    allocateNode(boundary);

    for (int i = 0; i < static_cast<int>(particles.size()); i++)
    {
        if (boundary.contains(particles[i].position))
            insert(0, i, particles);
    }
}

int QuadTree::allocateNode(AABB boundary)
{
    QuadNode node;
    node.boundary = boundary;
    nodes.push_back(node);
    return static_cast<int>(nodes.size()) - 1;
}

void QuadTree::subdivide(int nodeIndex)
{
    float half = nodes[nodeIndex].boundary.halfSize / 2.f;
    float cx = nodes[nodeIndex].boundary.x;
    float cy = nodes[nodeIndex].boundary.y;

    AABB regions[4] = {
        { cx - half, cy - half, half },
        { cx + half, cy - half, half },
        { cx - half, cy + half, half },
        { cx + half, cy + half, half }
    };

    for (int i = 0; i < 4; i++)
        nodes[nodeIndex].children[i] = allocateNode(regions[i]);
}

void QuadTree::insert(int nodeIndex, int particleIndex,
    const std::vector<Particle>& particles)
{
    // Iterative — uses heap stack, safe under MPI's small thread stack
    struct Task { int nodeIndex; int particleIndex; };

    std::vector<Task> stack;
    stack.reserve(64);
    stack.push_back({ nodeIndex, particleIndex });

    while (!stack.empty())
    {
        Task task = stack.back();
        stack.pop_back();

        int nIdx = task.nodeIndex;
        int pIdx = task.particleIndex;

        float        pm = particles[pIdx].mass;
        sf::Vector2f pp = particles[pIdx].position;
        float        newMass = nodes[nIdx].totalMass + pm;

        nodes[nIdx].centerOfMass =
            (nodes[nIdx].centerOfMass * nodes[nIdx].totalMass + pp * pm)
            / newMass;
        nodes[nIdx].totalMass = newMass;

        if (nodes[nIdx].isEmpty())
        {
            nodes[nIdx].particleIndex = pIdx;
            continue;
        }

        if (nodes[nIdx].isLeaf())
        {
            int existing = nodes[nIdx].particleIndex;
            nodes[nIdx].particleIndex = -1;
            subdivide(nIdx);

            for (int c = 0; c < 4; c++)
            {
                int childIdx = nodes[nIdx].children[c];
                if (childIdx != -1 &&
                    nodes[childIdx].boundary.contains(
                        particles[existing].position))
                {
                    stack.push_back({ childIdx, existing });
                    break;
                }
            }
        }

        for (int c = 0; c < 4; c++)
        {
            int childIdx = nodes[nIdx].children[c];
            if (childIdx != -1 &&
                nodes[childIdx].boundary.contains(pp))
            {
                stack.push_back({ childIdx, pIdx });
                break;
            }
        }
    }
}

sf::Vector2f QuadTree::computeForce(int particleIndex,
    const std::vector<Particle>& particles,
    float G, float epsilon) const
{
    if (nodes.empty()) return sf::Vector2f(0.f, 0.f);
    return forceFromNode(0, particleIndex, particles, G, epsilon);
}

sf::Vector2f QuadTree::forceFromNode(int nodeIndex, int particleIndex,
    const std::vector<Particle>& particles,
    float G, float epsilon) const
{
    const QuadNode& node = nodes[nodeIndex];

    if (node.totalMass == 0.f)
        return sf::Vector2f(0.f, 0.f);

    if (node.isLeaf() && node.particleIndex == particleIndex)
        return sf::Vector2f(0.f, 0.f);

    sf::Vector2f diff = node.centerOfMass - particles[particleIndex].position;
    float        distSq = diff.x * diff.x + diff.y * diff.y;
    float        dist = std::sqrt(distSq + epsilon * epsilon);

    float nodeWidth = node.boundary.halfSize * 2.f;
    if (node.isLeaf() || (nodeWidth / dist) < THETA)
    {
        float forceMag = G * particles[particleIndex].mass * node.totalMass
            / (distSq + epsilon * epsilon);
        return (forceMag / dist) * diff;
    }

    sf::Vector2f total(0.f, 0.f);
    for (int c = 0; c < 4; c++)
    {
        if (node.children[c] != -1)
            total += forceFromNode(node.children[c], particleIndex,
                particles, G, epsilon);
    }
    return total;
}