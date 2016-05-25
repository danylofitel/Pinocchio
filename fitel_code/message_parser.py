import glob
import re

__author__ = 'danylofitel'


def get_features(filename):
    features = []
    for i, line in enumerate(open(filename)):
        features.append(int(line))
    return features


def is_legitimate_message_file(filename):
    return "legit" in filename


def get_data(features, directory):
    regex = re.compile("\d+")

    spam_messages = []
    legitimate_messages = []

    for filename in glob.glob(directory):
        message_is_legitimate = is_legitimate_message_file(filename)
        words = {}

        for i, line in enumerate(open(filename)):
            for match in regex.findall(line):
                words[int(match)] = None

        feature_vector = []
        for feature in features:
            feature_vector.append(1 if feature in words else 0)

        if message_is_legitimate:
            legitimate_messages.append(feature_vector)
        else:
            spam_messages.append(feature_vector)

    return legitimate_messages, spam_messages


features = get_features("C:\\spam\\features.txt")
training_data = get_data(features, "C:\\spam\\training\\*.txt")
test_data = get_data(features, "C:\\spam\\testing\\*.txt")
