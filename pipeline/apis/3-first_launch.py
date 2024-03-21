#!/usr/bin/env python3
"""Write a script that displays the first launch with these information:

Name of the launch
The date (in local time)
The rocket name
The name (with the locality) of the launchpad
Format:
<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)

we encourage you to use the date_unix for sorting it - and if 2 launches have
the same date, use the first one in the API result.

Your code should not be executed when the file is imported
(you should use if __name__ == '__main__':)"""
import requests


if __name__ == '__main__':
    launches = requests.get(
        'https://api.spacexdata.com/v4/launches/upcoming').json()
    unix_dates = [launch['date_unix'] for launch in launches]
    min_idx = unix_dates.index(min(unix_dates))
    upcoming_launch = launches[min_idx]

    rocket = requests.get(
        'https://api.spacexdata.com/v4/rockets/{}'.format(
            upcoming_launch['rocket'])).json()
    launchpad = requests.get(
        'https://api.spacexdata.com/v4/launchpads/{}'.format(
            upcoming_launch['launchpad'])).json()

    print('{} ({}) {} - {} ({})'.format(upcoming_launch['name'],
                                        upcoming_launch['date_local'],
                                        rocket['name'],
                                        launchpad['name'],
                                        launchpad['locality']))
