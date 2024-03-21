#!/usr/bin/env python3
"""Write a script that displays the number of launches per rocket.

Use this https://api.spacexdata.com/v3/launches to make request
All launches should be taking in consideration
Each line should contain the rocket name and the number of launches separated
by : (format below in the example)
Order the result by the number launches (descending)
If multiple rockets have the same amount of launches, order them by
alphabetic order (A to Z)
Your code should not be executed when the file is imported
(you should use if __name__ == '__main__':)"""
import requests


if __name__ == '__main__':
    rockets = dict()
    launches = requests.get(
        'https://api.spacexdata.com/v3/launches').json()
    for launch in launches:
        if launch['rocket']['rocket_name'] not in rockets.keys():
            rockets[launch['rocket']['rocket_name']] = 1
        else:
            rockets[launch['rocket']['rocket_name']] += 1

    sorted_rockets = sorted(rockets.items(),
                            key=lambda kv: kv[1],
                            reverse=True)

    for sorted_rocket in sorted_rockets:
        print('{}: {}'.format(sorted_rocket[0], sorted_rocket[1]))
