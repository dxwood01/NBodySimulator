#include "Particle.h"
#include <cmath>

Particle::Particle(sf::Vector2f pos, sf::Vector2f vel, float r, float m)
    : position(pos),
    velocity(vel),
    radius(r),
    mass(m),
    force(0.f, 0.f),
    color(sf::Color::Green)
{
}

void Particle::update(float dt)
{
    sf::Vector2f accel = force / mass;
    velocity += accel * dt;
    position += velocity * dt;

    // Reset force after integration
    force = sf::Vector2f(0.f, 0.f);
}