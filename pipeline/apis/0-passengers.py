#!/usr/bin/env python3
"""Can I join? Module"""

import requests


def availableShips(passengerCount):
    """Create a method that returns the list of ships that can hold a given
    number of passengers:

        Prototype: def availableShips(passengerCount):
        Don’t forget the pagination
        If no ship available, return an empty list."""

    correct_ships = []
    pagination = 'https://swapi-api.hbtn.io/api/starships'
    while pagination:
        ships = requests.get(pagination).json()
        for starship in ships['results']:
            if starship["passengers"] == "unknown" or \
                    starship["passengers"] == "n/a":
                continue
            if int(starship["passengers"].replace(',', '')) >= passengerCount:
                correct_ships.append(starship['name'])
        pagination = ships['next']

    return correct_ships
