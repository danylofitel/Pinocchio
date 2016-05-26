import glob
import re

__author__ = 'danylofitel'


def get_filenames(directory):
    for filename in glob.glob(directory):
        yield filename


def is_legitimate_message_file(filename):
    return "legit" in filename


class MessageParser:
    def __init__(self, features_filename):
        self.encoded_word_regex = re.compile("\d+")
        self.features = []
        for i, line in enumerate(open(features_filename)):
            self.features.append(int(line))

    def feature_count(self):
        return len(self.features)

    def extract_feature_vector(self, filename):
        words = {}
        for i, line in enumerate(open(filename)):
            for match in self.encoded_word_regex.findall(line):
                words[int(match)] = None

        feature_vector = []
        for feature in self.features:
            feature_vector.append(1 if feature in words else 0)

        return feature_vector

    def extract_feature_vectors(self, directory):
        spam_messages = []
        legitimate_messages = []

        for filename in get_filenames(directory):
            feature_vector = self.extract_feature_vector(filename)

            if is_legitimate_message_file(filename):
                legitimate_messages.append(feature_vector)
            else:
                spam_messages.append(feature_vector)

        return legitimate_messages, spam_messages


'''
parser = MessageParser("C:\\spam\\features.txt")
training_data = parser.extract_feature_vectors("C:\\spam\\training\\*.txt")
test_data = parser.extract_feature_vectors("C:\\spam\\testing\\*.txt")
print len(training_data[0]) + len(training_data[1])
print len(test_data[0]) + len(test_data[1])
'''