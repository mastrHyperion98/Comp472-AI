# -------------------------------------------------------
# Assignment 1
# Written by Steven Smith 40057065
# For COMP 472 Section KX - Summer 2020
# -------------------------------------------------------

### Development Environment ###

Operating System: Windows 10 1909
IDE Used: Pycharm Pro
Virtual Environment Used: Conda/Anaconda


### Libraries Used ### 

geopandas
numpy
math
matplotlib
pandas
time

### Instructions for Pycharm ### 

1: Copy & Paste the data folder, main.py and functions.py into a Pycharm project. 
2: Place the cursor over File on the top left of the applications toolbar and click on it. 
3: Move your cursor down the drop down menu until you find settings and click on it.
4: Go down to Project:(Project Name) and select the Project Interpreter subtab. 
5: This will open up a table of the tools and librarties installed. On the right there is a + sign.
6: Click on it and in the search bar recursively search for all the libraries (math&time are native libraries)
7: Setup a new Python Run/Debug configuration for main.py


### Program Usage ### 

The program will prompt the user to input data such as the threshold, step size etc. The program assumes that all user input is of the right format. 
The prompt for Start and Goal are the only one with validation for point position to make sure the position is valid. The position must be inputed as a tuple - such as x,y. 
Anything else will throw an input error. 

Following the inputs the program will execute and display updates when it transitions from one process to another. Please be patient while the program performs
data wrangling and feature-engineering on the dataframes. This allows us to quickly fetch entire lists of data that we will use in our search algorithm.

When the program terminates it will output a "Program has terminated successfully" failure to see this message means an error has occured somewhere. 

### Heuristic Function ### 

A* Star requires a heuristic function to be admissible for it to be considered an A* search. Our heuristic function uses manhattan distance as our estimator. This is 
valid and admissible because the heuristic function assumes the cost will be lower than it actually is. Essentially it is the absolute value of the differance in position
components between our current position and the goal.