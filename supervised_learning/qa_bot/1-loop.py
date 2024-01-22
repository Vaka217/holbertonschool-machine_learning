#!/usr/bin/env python3
"""Takes in input from the user with the prompt Q: and prints A: as a response.
If the user inputs exit, quit, goodbye, or bye, case insensitive,
print A: Goodbye and exit."""


while 1:
    Q = input("Q: ")
    if Q.lower() in ["exit", "quit", "goodbye", "bye"]:
        print("A: Goodbye")
        break
    else:
        print("A: ")
