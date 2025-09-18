'''
Calvin Schaul
IDSN 542, Fall 2025
cschaul@usc.edu
Lab 4
'''
import random

FACTOR_LOWER_RANGE = 2
FACTOR_UPPER_RANGE = 59
NUMBER_OF_NUMBERS = 50000

def get_user_number():
    """ Ask user for an integer"""
    while True:
        try:
            user_number = int(input("Please enter a positive integer: "))
            if user_number <= 0:
                raise ValueError("Number must be positive.")
            return user_number
        except ValueError as ve:
            print("Invalid input:", ve)

def generate_n_numbers_0_to_50000(n):
    """ Generate a list of n random integers between 0 and 50000"""
    number_list = [random.randrange(NUMBER_OF_NUMBERS + 1) for number in range(n)]
    return list(set(number_list)) # Remove duplicates (unlikely but possible)


def calculate_factors_2_to_59(number_list):
    """ For each number in a list, calculate its factors between 2 and 59. Store the results in a dictionary."""
    factors_dict = {i: 0 for i in range(FACTOR_LOWER_RANGE, FACTOR_UPPER_RANGE + 1)}
    for number in number_list:
        for factor in factors_dict:
            if number % factor == 0:
                factors_dict[factor] += 1
    return factors_dict

def print_factors_dict(factors_dict):
    """ Print the factors dictionary in a readable format."""
    for factor, count in factors_dict.items():
        if count > 0:
            a = "*" * count
            print(f"{factor:<2d} : {a}")

def main():
    user_number = get_user_number()
    number_list = generate_n_numbers_0_to_50000(user_number)
    factors_dict = calculate_factors_2_to_59(number_list)
    print_factors_dict(factors_dict)

if __name__ == "__main__":
    main()