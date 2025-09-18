import random

words = [
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

alphabet_as_list = [chr(i) for i in range(ord('a'), ord('z') + 1)] # Love list comprehensions


# Wrote this on an airplane without internet.
def jumbleWord(word):
    # Create a list from the letters in our word
    word_as_list = [letter for letter in word]

    # Create a new list and randomly pop letters from the original list into the new list
    jumbled_word_as_list = []
    while len(word_as_list) > 0:
        jumbled_word_as_list.append(word_as_list.pop(random.randint(0,len(word_as_list) - 1)))

    # Return the jumbled string
    return ''.join(jumbled_word_as_list)

def jumbleGame():
    guessCount = 1 # User will always have at least one guess
    word = random.choice(words)
    jumbledWord = jumbleWord(word)

    print("The jumbled word is \"" + str(jumbledWord) + "\"")
    
    # Main Game Loop
    while True:
        # Validate Input: User can only enter alphabetic characters
        try:
            userGuess = input("Please enter your guess: ")
            if userGuess.isalpha() == False:
                # Invalid Input, raise ValueError and increment guess! No mercy!
                guessCount += 1
                raise ValueError("Input must only contain letters.")
            elif userGuess.lower() != word:
                # Incorrect Alphabetic Guess
                guessCount += 1
                print("Try again.\n")
            else:
                # Corect Guess
                break
        except ValueError as ve:
            print(str(ve) + "\n")
            continue

    # Exited loop, user must have guessed correctly
    print("You got it!")
    if guessCount == 1:
        print("It took you 1 try.")
    else:
        print("It took you " + str(guessCount) + " tries.")

def encryptMessage(message, shift):
    encrypted_message = []
    for char in message:
        if char not in alphabet_as_list:
            encrypted_message.append(char) # Non-alphabetic characters are not encrypted/decrypted
        else:
            base_index = alphabet_as_list.index(char)
            shift_index = (base_index + shift) % 26 # Add shift to encrypt
            encrypted_message.append(alphabet_as_list[shift_index])
    
    return ''.join(encrypted_message) # Return the encrypted message as a string

def decryptMessage(message, shift):
    decrypted_message = []
    for char in message:
        if char not in alphabet_as_list:
            decrypted_message.append(char) # Non-alphabetic characters are not encrypted/decrypted
        else:
            base_index = alphabet_as_list.index(char)
            shift_index = (base_index - shift) % 26 # Subtract shift to decrypt
            decrypted_message.append(alphabet_as_list[shift_index])

    return ''.join(decrypted_message) # Return the decrypted message as a string

def caesarCipherGame():
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
            encrypted_message = encryptMessage(user_message, shift_amount)
            print("\n\tEncrypted message: " + encrypted_message)

            print("\nDecrypting message...")
            decrypted_message = decryptMessage(encrypted_message, shift_amount)
            print("\n\tDecrypted message: " + decrypted_message)
            print("\n\tOriginal Message: " + user_message)
    except ValueError as ve:
        print(str(ve))
        return #Should this just call casearCipherGame() again?
    

def main():
    print("Starting Jumble Game...\n")
    jumbleGame()
    print("\nStarting Caesar Cipher Game...\n")
    caesarCipherGame()

if __name__ == "__main__":
    main()