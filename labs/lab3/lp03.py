'''
Calvin Schaul
IDSN 542, Fall 2025
cschaul@usc.edu
Lab 3
'''

# List of sci-fi curse words to censor and the desired replacement
curse_words = ["gren", "gral", "drel", "fron", "glud", "xarp", "nark"]
censored_replacement = "BEEP"

# Function to censor sci-fi curse words from a sentence
def sciFiCensor(sentence):
    #Convert string input into list
    words = sentence.split(' ')
    for i in range(len(words)):
        # Replace bad word with censor word in list
        if words[i].lower() in curse_words:
            words[i] = censored_replacement


    # Print the censored sentence from the cenesored list (sentence)
    print(' '.join(words), end='')

def main():
    #Prompt user for a sentence to censor
    userSentence = input("What shall I censor: ")
    
    #Continue censoring sentences until an empty string is entered
    while userSentence != "":
        sciFiCensor(userSentence)
        print() #Print newline at 
        userSentence = input("What shall I censor: ")

    #After recieved an empty string, exit program
    print("Goodbye!")

if __name__ == "__main__":
    main()