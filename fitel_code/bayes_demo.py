from message_parser import MessageParser
from naive_bayes import NaiveBayes
from semi_naive_bayes import SemiNaiveBayes

__author__ = 'danylofitel'


features_filename = "C:\\spam\\features.txt"
training_directory = "C:\\spam\\training\\*.txt"
testing_directory = "C:\\spam\\testing\\*.txt"

parser = MessageParser(features_filename)
training_set = parser.extract_feature_vectors(training_directory)
testing_set = parser.extract_feature_vectors(testing_directory)

naive_bayes = SemiNaiveBayes(parser.feature_count(), testing_set[0], testing_set[1])

false_positives = 0
for legitimate in testing_set[0]:
    if naive_bayes.is_spam(legitimate):
        false_positives += 1

print "False positives: ", false_positives

false_negatives = 0
for spam in testing_set[1]:
    if not naive_bayes.is_spam(spam):
        false_negatives += 1

print "False negatives: ", false_negatives
