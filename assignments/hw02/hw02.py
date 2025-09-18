'''
Calvin Schaul
IDSN 542, Fall 2025
cschaul@usc.edu
Homework 2
'''

import random

WORDS = [
    "trojan",
    "iovine",
    "young",
    "california",
    "machine",
    "intelligence",
    "python",
    "program",
    "compiler",
    "editor"
]

ALPHABET_AS_LIST = list('abcdefghijklmnopqrstuvwxyz')

# Wrote this on an airplane without internet.
def jumble_word(word):
    """ Jumbles the letters in a given word and returns the jumbled string."""
    word_as_list = [letter for letter in word]

    # Create a new list and randomly pop letters from the original list into the new list
    jumbled_word_as_list = []
    while len(word_as_list) > 0:
        jumbled_word_as_list.append(word_as_list.pop(random.randint(0,len(word_as_list) - 1)))

    # Return the jumbled string
    return ''.join(jumbled_word_as_list)

def jumble_game():
    """ Plays a simple jumble game with the user."""
    guess_count = 1 # User will always have at least one guess
    word = random.choice(WORDS)
    jumbled_word = jumble_word(word)

    print("The jumbled word is \"" + str(jumbled_word) + "\"")
    
    # Main Game Loop
    while True:
        # Validate Input: User can only enter alphabetic characters
        try:
            user_guess = input("Please enter your guess: ")
            if user_guess.isalpha() == False:
                # Invalid Input, raise ValueError and increment guess! No mercy!
                guess_count += 1
                raise ValueError("Input must only contain letters.")
            elif user_guess.lower() != word:
                # Incorrect Alphabetic Guess
                guess_count += 1
                print("Try again.\n")
            else:
                # Corect Guess
                break
        except ValueError as ve:
            print(str(ve) + "\n")
            continue

    # Exited loop, user must have guessed correctly
    print("You got it!")
    if guess_count == 1:
        print("It took you 1 try.")
    else:
        print("It took you " + str(guess_count) + " tries.")

def encrypt_message(message, shift):
    """ Encrypts a message using a Caesar cipher with the given shift amount."""
    encrypted_message = []
    for char in message:
        if char not in ALPHABET_AS_LIST:
            encrypted_message.append(char) # Non-alphabetic characters are not encrypted/decrypted
        else:
            base_index = ALPHABET_AS_LIST.index(char)
            shift_index = (base_index + shift) % 26 # Add shift to encrypt
            encrypted_message.append(ALPHABET_AS_LIST[shift_index])
    
    return ''.join(encrypted_message) # Return the encrypted message as a string

def decrypt_message(message, shift):
    """ Decrypts a message using a Caesar cipher with the given shift amount."""
    decrypted_message = []
    for char in message:
        if char not in ALPHABET_AS_LIST:
            decrypted_message.append(char) # Non-alphabetic characters are not encrypted/decrypted
        else:
            base_index = ALPHABET_AS_LIST.index(char)
            shift_index = (base_index - shift) % 26 # Subtract shift to decrypt
            decrypted_message.append(ALPHABET_AS_LIST[shift_index])

    return ''.join(decrypted_message) # Return the decrypted message as a string

def caesar_cipher_game():
    """ Plays a simple Caesar cipher game with the user."""
    # User can enter any messsage, we will just shift the alphabetic characters and leave the rest alone
    user_message = input("Enter a message: ")

    # Get and Validate Shift: Shift amount must be an integer between 0 and 25
    try:
        shift_amount = input("\nEnter a shift amount (0-25): ")
        if not shift_amount.isdigit():
            raise ValueError("Shift amount must be an integer between 0 and 25.")
        elif int(shift_amount) < 0 or int(shift_amount) > 25:
            raise ValueError("Shift amount must be between 0 and 25.")
        else:
            # Requirements don't specify case-sensitivity, so just encrypt lowercase
            shift_amount = int(shift_amount)
            user_message = user_message.lower()
            
            # Encrypt and Decrypt
            print("\nEncrypting message...")
            encrypted_message = encrypt_message(user_message, shift_amount)
            print("\n\tEncrypted message: " + encrypted_message)

            print("\nDecrypting message...")
            decrypted_message = decrypt_message(encrypted_message, shift_amount)
            print("\n\tDecrypted message: " + decrypted_message)
            print("\n\tOriginal Message: " + user_message)
    except ValueError as ve:
        print(str(ve) + "\n")
        caesar_cipher_game()
    

def main():
    print("Starting Jumble Game...\n")
    jumble_game()
    print("\nStarting Caesar Cipher Game...\n")
    caesar_cipher_game()

if __name__ == "__main__":
    main()