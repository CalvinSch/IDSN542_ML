'''
Calvin Schaul
IDSN 542, Fall 2025
cschaul@usc.edu
Lab 2
'''

#Prompt user for their name
userInputName = input("What's your name (enter to quit)? ")

#After user enters their name, analyze one letter at a time
while len(userInputName) > 0:
    #Keep track of vowels and x's in the input name
    countAEIOUX = 0

    for letter in userInputName.lower():
        if letter in 'aeioux':
            countAEIOUX += 1
    
    #Print vowel and x count - account for singular vowel
    if countAEIOUX == 1:
        print("That name has " + str(countAEIOUX) + " vowel!")
    else:
        print("That name has " + str(countAEIOUX) + " vowels!")

    #Reset count and get input again
    countAEIOUX = 0
    userInputName = input("What's your name? (enter to quit) ")