'''
Calvin Schaul
IDSN 542, Fall 2025
cschaul@usc.edu
Lab 1
'''

#Greet the user and list four specials
print("Welcome to Calvin's Restaurant, how may I take your order?")
print("We have the following specials today:")
print("1. All-Day Breakfast")
print("2. Taco Bar")
print("3. Off-The-Grill")
print("4. Seafood Platter")

#Prompt the user to select a special
print("Please enter the number of the special you would like to order:")
special = input()

print()

if special == '1':
    print("Pick one of the following food options: 1. eggs 2. bacon 3. toast")
    print("Please enter the food option you would like:")
    food_option = input().lower()

    #Print user's food option
    print("You have selected the All-Day Breakfast special with: ", end='\t')

    #Print the food option based on user input
    if food_option == 'eggs': print("eggs")
    elif food_option == 'bacon': print("bacon")
    elif food_option == 'toast': print("toast")
    else: print("Invalid selection. Please restart your order and select a valid food option.")
elif special == '2':
    print("Pick one of the following food options: 1. carne asada 2. chicken 3. al pastor")
    print("Please enter the food option you would like:")
    food_option = input().lower()

    print("You have selected the Taco Bar special with: ", end='\t')

    if food_option == 'carne asada': print("carne asada")
    elif food_option == 'chicken': print("chicken")
    elif food_option == 'al pastor': print("al pastor")
    else: print("Invalid selection. Please restart your order and select a valid food option.") 
elif special == '3':
    print("Pick one of the following food options: 1. steak 2. chicken 3. salmon")
    print("Please enter the food option you would like:")
    food_option = input().lower()

    print("You have selected the Off-The-Grill special with:", end='\t')

    if food_option == 'steak': print("steak")
    elif food_option == 'chicken': print("chicken")
    elif food_option == 'salmon': print("salmon")
    else: print("Invalid selection. Please restart your order and select a valid food option.")

elif special == '4':
    print("Pick one of the following food options: 1. shrimp 2. crab 3. lobster")
    print("Please enter the food option you would like:")
    food_option = input().lower()

    print("You have selected the Seafood Platter special with: ", end='\t')

    if food_option == 'shrimp': print("shrimp")
    elif food_option == 'crab': print("crab")
    elif food_option == 'lobster': print("lobster")
    else: print("Invalid selection. Please restart your order and select a valid food option.")
else:
    print("Invalid selection. Please restart your order and select a valid special.")
