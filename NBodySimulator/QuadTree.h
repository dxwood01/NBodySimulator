#pragma once
#include <SFML/System/Vector2.hpp>
#include <vector>
#include "Particle.h"

struct AABB
{
    float x, y;
    float halfSize;

    bool contains(sf::Vector2f pos) const
    {
        return pos.x >= x - halfSize && pos.x <= x + halfSize &&
            pos.y >= y - halfSize && pos.y <= y + halfSize;
    }
};

struct QuadNode
{
    AABB         boundary;
    float        totalMass;
    sf::Vector2f centerOfMass;
    int          children[4];
    int          particleIndex;

    bool isLeaf()  const { return children[0] == -1; }
    bool isEmpty() const { return particleIndex == -1 && isLeaf(); }

    QuadNode() : totalMass(0.f), centerOfMass(0.f, 0.f), particleIndex(-1)
    {
        children[0] = children[1] = children[2] = children[3] = -1;
    }
};

class QuadTree
{
public:
    static constexpr float THETA = 0.5f;

    std::vector<QuadNode> nodes;

    void         build(const std::vector<Particle>& particles, AABB boundary);
    sf::Vector2f computeForce(int particleIndex,
        const std::vector<Particle>& particles,
        float G, float epsilon) const;

private:
    void         insert(int nodeIndex, int particleIndex,
        const std::vector<Particle>& particles);
    void         subdivide(int nodeIndex);
    sf::Vector2f forceFromNode(int nodeIndex, int particleIndex,
        const std::vector<Particle>& particles,
        float G, float epsilon) const;
    int          allocateNode(AABB boundary);
};