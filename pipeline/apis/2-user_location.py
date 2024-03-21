#!/usr/bin/env python3
import sys
import requests


if __name__ == '__main__':
    user = requests.get(sys.argv[1])
    if user.status_code == 404:
        print('Not found')
    elif user.status_code == 403:
        print('Reset in {} min'.format(user.headers))
    else:
        user = user.json()
        print(user['location'])
