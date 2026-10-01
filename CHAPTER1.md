1\. Write a program to print Twinkle twinkle little star poem in python.

&#x20;print("""Twinkle, twinkle, little star,

How I wonder what you are!

Up above the world so high,

Like a diamond in the sky.



Twinkle, twinkle, little star,

How I wonder what you are!""")



2\. Use REPL and print the table of 5 using it.

>>> 5 \* 1

5

>>> 5 \* 2

10

>>> 5 \* 3

15

>>> 5 \* 4

20

>>> 5 \* 5

25

>>> 5 \* 6

30

>>> 5 \* 7

35

>>> 5 \* 8

40

>>> 5 \* 9

45

>>> 5 \* 10

50

3\. Install an external module and use it to perform an operation of your interest.

pip install pyjokes



import pyjokes

print(pyjokes.get\_joke())



4\. Write a python program to print the contents of a directory using the os module. Search

online for the function which does that.

import os



path = "."  # Current directory



contents = os.listdir(path)



print("Contents of the directory:")

for item in contents:

&#x20;   print(item)



5\. Label the program written in problem 4 with comments.

\# Import the os module

import os



\# Specify the directory path

\# "." means the current working directory

path = "."



\# Get the list of files and folders in the directory

contents = os.listdir(path)



\# Print a heading

print("Contents of the directory:")



\# Loop through each item and print it

for item in contents:

&#x20;   print(item)





