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

# Wrote this on an airplane without internet.
def jumbleWord(word):
    # Create a list from the letters in our word
    word_as_list = [letter for letter in word]

    # Create a new list and randomly pop letters from the original list into the new list
    jumbled_word_as_list = []
    while len(word_as_list) > 0:
        jumbled_word_as_list.append(word_as_list.pop(random.randint(0,len(word_as_list) - 1)))

    # Return the jumbled string
    #print(jumbled_word_as_list)
    return ''.join(jumbled_word_as_list)

def jumbleGame():
    guessCount = 1 # User will always have at least one guess
    word = random.choice(words)
    jumbledWord = jumbleWord(word)

    print("The jumbled word is \"" + str(jumbledWord) + "\"")
    
    #Main Game Loop
    while True:
        #Validate Input: User can only enter alphabetic characters
        try:
            userGuess = input("Please enter your guess: ")
            if userGuess.isalpha() == False:
                #Invalid Input, raise ValueError and increment guess! No mercy!
                guessCount += 1
                raise ValueError("Input must only contain letters.")
            elif userGuess.lower() != word:
                #Incorrect Alphabetic Guess
                guessCount += 1
                print("Try again.\n")
            else:
                #Corect Guess
                break
        except ValueError as ve:
            print(str(ve) + "\n")
            continue
    # userGuess = input("Please enter your guess: ")

    # while userGuess.lower() != word:
    #     guessCount += 1
    #     print("Try again.\n")
    #     userGuess = input("Please enter your guess: ")

    # Exited loop, user must have guessed correctly
    print("You got it!")
    if guessCount == 1:
        print("It took you 1 try.")
    else:
        print("It took you " + str(guessCount) + " tries.")

def main():
    jumbleGame()

if __name__ == "__main__":
    main()