#pragma once
#include <SFML/System/Vector2.hpp>
#include <SFML/Graphics/Color.hpp>

class Particle
{
public:
    sf::Vector2f position;
    sf::Vector2f velocity;
    float radius;
    float mass;          // ✅ new
    sf::Vector2f force;  // ✅ new, replaces acceleration
    sf::Color color;

    Particle(sf::Vector2f pos, sf::Vector2f vel, float r, float m);

    void update(float dt);
};