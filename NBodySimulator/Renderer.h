#pragma once
#include <SFML/Graphics.hpp>
#include <vector>
#include "Particle.h"

class Renderer
{
public:
    void draw(sf::RenderWindow& window, const std::vector<Particle>& particles);
};