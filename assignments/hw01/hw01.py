'''
Calvin Schaul
IDSN 542, Fall 2025
cschaul@usc.edu
Homework 1
'''

## Part 1: Prompt for Input

#Prompt user for their name
name = input("Hello, whose BMI shall I calculate? ")

#Loop until empty string is entered for name
while len(name) != 0:
    #Prompt user for height information
    print("Okay first I need " + name + "'s height. I'll take it in feet and inches.")
    userHeightFeet = int(input("Feet first... "))
    userHeightInches = int(input("Now inches... "))

    #Prompt user for weight information. Store pounds as a float
    print("Thanks. Now I need " + name + "'s weight in pounds.")
    userWeightPounds = float(input("Please enter " + name + "'s weight... "))

    ## Part 2: Calculate Total Inches

    totalHeightInches = (userHeightFeet * 12) + userHeightInches

    ## Part 3: Calculate Height in Meters

    totalHeightMeters = totalHeightInches / 39.37

    ## Part 4: Calculate Mass in Kilograms

    userWeightKilograms = userWeightPounds / 2.2

    ## Part 5: Calculate and Output Final BMI

    finalBMI = userWeightKilograms / (totalHeightMeters ** 2)
    print(name + "'s BMI is " + str(round(finalBMI, 1)) + ".")

    #Prompt for next name
    name = input("\nHello, whose BMI shall I calculate? ")
