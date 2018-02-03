# import os
# from time import sleep

# # Greeter is a terminal application that greets old friends warmly,
# #   and remembers new friends.


# ### FUNCTIONS ###

# def display_title_bar():
#     # Clears the terminal screen, and displays a title bar.
#     os.system('clear')
              
#     print("\t**********************************************")
#     print("\t***  Greeter - Hello old and new friends!  ***")
#     print("\t**********************************************")
    

# ### MAIN PROGRAM ###    

# # Print a bunch of information, in short intervals
# names = ['aaron', 'brenda', 'cyrene', 'david', 'eric']

# # Print each name 5 times.
# for name in names:
#     display_title_bar()

#     print("\n\n")
#     for x in range(0,5):
#         print(name.title())
    
#     # Pause for 1 second between batches.
#     sleep(1)





#!/usr/bin/env python
# encoding: utf-8

import npyscreen

def myFunction(*args):
    F = npyscreen.Form(name='My Test Application')
    F.edit()

if __name__ == '__main__':
    npyscreen.wrapper_basic(myFunction)
    print "Blink and you missed it!"