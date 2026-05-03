#include "Renderer.h"

void Renderer::draw(sf::RenderWindow& window, const std::vector<Particle>& particles)
{
    for (const auto& p : particles)
    {
        sf::CircleShape circle(p.radius);
        circle.setOrigin({ p.radius, p.radius });
        circle.setPosition(p.position);
        circle.setFillColor(p.color);

        window.draw(circle);
    }
}