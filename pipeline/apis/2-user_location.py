#!/usr/bin/env python3
"""By using the GitHub API, write a script that prints the location of a
specific user:

The user is passed as first argument of the script with the full API URL,
example: ./2-user_location.py https://api.github.com/users/holbertonschool
If the user doesn’t exist, print Not found
If the status code is 403, print Reset in X min where X is the number of
minutes from now and the value of X-Ratelimit-Reset
Your code should not be executed when the file is imported (you should use
if __name__ == '__main__':)"""
import sys
import requests
import time


if __name__ == '__main__':
    if len(sys.argv) != 2:
        exit()
    user = requests.get(sys.argv[1])
    if user.status_code == 404:
        print('Not found')
    elif user.status_code == 403:
        reset_time = int(user.headers['X-Ratelimit-Reset'])
        current_time = int(time.time())
        time_difference = int((reset_time - current_time) / 60)
        print('Reset in {} min'.format(time_difference))
    else:
        user = user.json()
        print(user['location'])
