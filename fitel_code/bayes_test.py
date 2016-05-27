from message_parser import MessageParser
from naive_bayes import NaiveBayes
from non_naive_bayes import NonNaiveBayes

__author__ = 'danylofitel'


features_filename = "C:\\spam\\features_short.txt"
training_directory = "C:\\spam\\training\\*.txt"
testing_directory = "C:\\spam\\testing\\*.txt"

parser = MessageParser(features_filename)
training_set = parser.extract_feature_vectors(training_directory)
testing_set = parser.extract_feature_vectors(testing_directory)

bayesian_classifier = NaiveBayes(parser.feature_count(), training_set[0], training_set[1])

N_L = len(testing_set[0])
N_S = len(testing_set[1])
N = N_L + N_S

N_L_S = 0  # false positives
for legitimate in testing_set[0]:
    if bayesian_classifier.is_spam(legitimate):
        N_L_S += 1

N_S_L = 0  # false negatives
for spam in testing_set[1]:
    if not bayesian_classifier.is_spam(spam):
        N_S_L += 1

E = (N_S_L + N_L_S) / float(N)

P = 1.0 - E

F_L = N_L_S / float(N_L)
F_S = N_S_L / float(N_S)

print N_L_S
print N_S_L
print E
print P
print F_L
print F_S

print ""

print "N = ", N
print "N_L = ", N_L
print "N_S = ", N_S
print "N_L_S = ", N_L_S
print "N_S_L = ", N_S_L
print "E = ", E
print "P = ", P
print "F_L = ", F_L
print "F_S = ", F_S
