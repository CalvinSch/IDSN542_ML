"""
Calvin Schaul
IDSN 542, Fall 2025
cschaul@usc.edu
Homework 3
"""

import os

# Months dictionary with number of days in each month
MONTHS_DICT = {
    "january": 31,
    "february": 28,
    "march": 31,
    "april": 30,
    "may": 31,
    "june": 30,
    "july": 31,
    "august": 31,
    "september": 30,
    "october": 31,
    "november": 30,
    "december": 31
}

# Dictionary with months as keys, and a list <MONTHS_DICT[month]> empty strings as values
CALENDAR_DICT = {month: [''] * MONTHS_DICT[month] for month in MONTHS_DICT}

def get_user_date_inputs():
    """ Prompts the user for a month and a day until empty input is given."""
    while True:
        try:
            # Get input and split into month and day, 
            user_input = input("\nEnter a date for a holiday (for example \"July 1\"): ")
            if user_input == "":
                return
            split_input = user_input.split(" ")
            if len(split_input) != 2:
                raise Exception()
            
            # Validate month
            month = split_input[0].lower()
            if month not in MONTHS_DICT:
                raise ValueError(f"I don't know the month \"{month}\"")
            
            # Validate day
            day = split_input[1]
            if not day.isdigit():
                raise ValueError("The day must be a number.")
            day = int(day)
            if day < 1 or day > MONTHS_DICT[month.lower()]:
                raise ValueError(f"That month has has {MONTHS_DICT[month]} days.")
            return month, day
        
        except ValueError as e:
            print(f"{e}\n")
        except Exception as e:
            print("I don't see good input in there!\n")

def add_event_to_calendar(month, day):
    """ Prompts the user for an event and adds it to the calendar dictionary."""
    prompt = f"What happens on {month.capitalize()}, {day}? "
    event = input(prompt)
    
    # If event is not empty, add it to the calendar dictionary
    if event != "":
        if CALENDAR_DICT[month][day - 1] != "":
            # Append to exsitng events! Gotta keep that busy schedule in tact
            CALENDAR_DICT[month][day - 1] += "; " + event
        else: 
            CALENDAR_DICT[month][day - 1] = event
    # Nothing entered, do nothing
    else: 
        print("No event entered, nothing added to calendar.")
    return

def write_calendar_to_file(filename):
    """ Writes the calendar dictionary to a file."""
    if filename == "":
        return
    try:
        with open(filename, "x") as file:
            for month in CALENDAR_DICT.keys():
                for day in range(len(CALENDAR_DICT[month])):
                    file.write(f"{month},{day},{CALENDAR_DICT[month][day]}\n")
    except FileExistsError:
        print("File already exists, overwriting...")
        with open(filename, "w") as file:
            for month in CALENDAR_DICT.keys():
                for day in range(len(CALENDAR_DICT[month])):
                    file.write(f"{month},{day},{CALENDAR_DICT[month][day]}\n")
    return

def read_calendar_from_file(filename):
    """ Read calendar from existing events texst file"""
    try:
        with open(filename, "r") as file:
            entry = file.readline()
            while entry != "":
                event_info = entry.split(",")
                print(event_info)
                event_month = event_info[0]
                event_date_index = int(event_info[1])
                event_description = event_info[2].strip('\n')

                CALENDAR_DICT[event_month][event_date_index] = event_description
                # print(entry)
                entry = file.readline()
    except FileNotFoundError:
        print("File not found.")
    
    display_calendar()
    return

def display_calendar():
    """ Display calendar dates with events"""
    # Loop through months and events and print
    print("\n") # Line space before dislpaying 
    for month in CALENDAR_DICT.keys():
        for event in CALENDAR_DICT[month]:
            if event != "": print(f"{month.capitalize()} {CALENDAR_DICT[month].index(event) + 1} : {event}")
    return

def main():
    """ Main Loop: Prompt to read from existing calendar text file, prompt for events, then display and save."""
    existing_calendar_input = input("Enter the file name to read your events: ")
    try:
        if existing_calendar_input != "":
            read_calendar_from_file(existing_calendar_input)
    except Exception as e:
        print(f"Could not read from file \"{existing_calendar_input}\": {e}")

    # Get and process user inputs until empty input is given
    while True:
        date_input = get_user_date_inputs()
        if date_input is None:
            break

        month, day = date_input
        add_event_to_calendar(month, day)
    
    #Display all events & prompt user to write to file
    display_calendar()
    calendar_filename = input("Enter the file name to save your events: ")
    write_calendar_to_file(calendar_filename)
    print("\nGoodbye!")

if __name__ == "__main__":
    main()