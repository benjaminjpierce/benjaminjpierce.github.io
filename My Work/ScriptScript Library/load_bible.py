import os
import re

def extract_book_name(line):
    """
        Extracts name of the book from given line

        Args:
        - line (str): line from Bible text

        Returns:
        - str or None: name of the book (if exists), None if not.
    """

    match = re.match(r'^(\d?[^\d]+)', line)
    if match:
        return match.group(1).strip()
    return None


def split_bible_into_books(input_file, output_folder, books_to_include):
    """
        Splits full Bible text into separate files for specified books

        Args:
        - input_file (str): path to the full Bible text file
        - output_folder (str): folder to save individual book files in
        - books_to_include (list): list of books to include

        Returns:
        - None
    """

    # create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # read content of full Bible file
    with open(input_file, 'r', encoding='utf-8') as file:
        bible_lines = file.readlines()

    current_book = None
    current_file = None

    for line in bible_lines:

        # check if line has numerical characters (only numerical characters are chapter/verse)
        if any(char.isdigit() for char in line):

            # extract book from line
            book = extract_book_name(line)

            # check if current book is in list of desired books to include
            if book in books_to_include:

                # rename Matthew file
                if book == 'Mat':
                    book = 'Matthew'

                # start new book (or continue the first one)
                if book != current_book:
                    current_book = book

                    # close current file (if any)
                    if current_file:
                        current_file.close()

                    # create new file for current book
                    current_file_path = os.path.join(output_folder, f"{current_book}.txt")
                    current_file = open(current_file_path, 'w', encoding='utf-8')

                # write the line to current file
                if current_file:
                    current_file.write(line)

    # close last file
    if current_file:
        current_file.close()


if __name__ == "__main__":

    input_file = 'bible.txt'
    output_folder = 'bible books'
    books_to_include = ['Mat', 'Mark', 'Luke', 'John']

    split_bible_into_books(input_file, output_folder, books_to_include)
