"""
Calvin Schaul
IDSN 542, Fall 2025
cschaul@usc.edu
Homework 3
"""

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
            user_input = input("Enter a date for a holiday (for example \"July 1\"): ")
            if user_input == "":
                return
            split_input = user_input.split(" ")
            if len(split_input) != 2:
                raise Exception()
            month = split_input[0]
            day = int(split_input[1])
            if month not in MONTHS_DICT:
                raise ValueError(f"I don't know the month \"{month}\"")
            if day < 1 or day > MONTHS_DICT[month.lower()]:
                raise ValueError(f"That month has has {MONTHS_DICT[month]} days.")
            return month, day
        except ValueError as e:
            print(f"{e}\n")
        except Exception as e:
            print(f"I don't see good input in there!\n")

def add_event_to_calendar(month, day, event):
    return

def write_calendar_to_file(filename):
    return

def read_calendar_from_file(filename):
    return

def main():
    read_calendar_from_file("calendar.txt")
    get_user_date_inputs()
    write_calendar_to_file("calendar.txt")

if __name__ == "__main__":
    main()

