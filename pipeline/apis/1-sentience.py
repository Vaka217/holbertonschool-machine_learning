#!/usr/bin/env python3
"""Where I am? Module"""

import requests


def sentientPlanets():
    """Returns the list of names of the home planets of all sentient species.

    sentient type is either in the classification or designation attributes."""

    homeworlds = set()
    sentient_planets = []
    pagination = 'https://swapi-api.hbtn.io/api/species'
    while pagination:
        species = requests.get(pagination).json()
        for specie in species['results']:
            if specie['homeworld'] is not None and \
                (specie['classification'] == 'sentient' or
                 specie['designation'] == 'sentient'):
                homeworlds.add(specie['homeworld'])
        pagination = species['next']

    pagination = 'https://swapi-api.hbtn.io/api/planets'
    while pagination:
        planets = requests.get(pagination).json()
        for planet in planets['results']:
            if planet['url'] in homeworlds:
                sentient_planets.append(planet['name'])
        pagination = planets['next']

    return sentient_planets
