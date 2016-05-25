import glob
import re
import semi_naive_bayes

__author__ = 'danylofitel'


def get_features():
    features_filename = "C:\\spam\\features.txt"

    features = []
    for i, line in enumerate(open(features_filename)):
        features.append(int(line))

    return features


features = get_features()


print features
