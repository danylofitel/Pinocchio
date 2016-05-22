import glob
import re

__author__ = 'danylofitel'


def get_training_data():
    regex = re.compile("\d*")

    words = []

    for filename in glob.glob("C:\\spam\\all\\*.txt"):
        for i, line in enumerate(open(filename)):
            for match in regex.findall(line):
                words.append(match.zfill(5))

    sorted_words = sorted(set(words))
    for w in sorted_words:
        print w

    return

    prev_number = -1
    for number in sorted_words:
        current_number = int(number)
        if current_number - prev_number != 1:
            print current_number
        prev_number = current_number


def get_test_data():
    return [], []

get_training_data()
