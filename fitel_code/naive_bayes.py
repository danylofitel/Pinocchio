import sys

__author__ = 'danylofitel'


class NaiveBayes:

    # Lambda = Loss(L, S) /  Loss(S,L) is the additional parameter that
    # specifies the risk of misclassifying legitimate messages as spam.
    # As the value of Lambda increases, the classifier produces fewer false positives.
    misclassification_risk = 1.0

    # The number of features in the feature vector
    n = 0

    # The number of training legitimate messages.
    legitimate_count = 0

    # The number of training spam messages.
    spam_count = 0

    # The total number of training messages.
    total_count = 0

    # lambda * P(L) / P(S) is the classification threshold
    threshold = 0.0

    # Lambda i(x) = P(xi, S) / P(x_i, L) is the likelihood ratio.
    likelihood_0 = []
    likelihood_1 = []

    def __init__(self, n, legitimate, spam):
        if n <= 0 or legitimate is None or spam is None:
            raise Exception("Invalid arguments")

        self.n = n
        self.likelihood_0 = [0 for i in range(0, self.n)]
        self.likelihood_1 = [0 for i in range(0, self.n)]

        self.legitimate_count = len(legitimate)
        self.spam_count = len(spam)
        self.total_count = self.legitimate_count + self.spam_count

        p_l = self.legitimate_count / float(self.total_count)
        p_s = self.spam_count / float(self.total_count)

        self.threshold = self.misclassification_risk * p_l / p_s

        self.train(legitimate, spam)

    def train(self, legitimate, spam):
        legitimate_counts = [0 for i in range(0, self.n)]
        spam_counts = [0 for i in range(0, self.n)]

        for message in legitimate:
            for i in range(0, self.n):
                legitimate_counts[i] += message[i]

        for message in spam:
            for i in range(0, self.n):
                spam_counts[i] += message[i]

        for i in range(0, self.n):
            probability_legitimate = legitimate_counts[i] / float(self.legitimate_count)
            probability_spam = spam_counts[i] / float(self.spam_count)

            self.likelihood_0[i] = self.divide((1.0 - probability_spam), (1.0 - probability_legitimate))
            self.likelihood_1[i] = self.divide(probability_spam, probability_legitimate)

    @staticmethod
    def divide(nom, den):
        if den == 0.0:
            if nom == 0.0:
                return 1.0
            else:
                return sys.float_info.max
        else:
            return nom / den

    def is_spam(self, vector):
        likelihood = 1.0
        for i in range(0, self.n):
            likelihood *= self.likelihood_0[i] if vector[i] == 0 else self.likelihood_1[i]

        return likelihood > self.threshold
