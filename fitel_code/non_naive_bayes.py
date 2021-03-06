import sys

__author__ = 'danylofitel'


class Node:
    def __init__(self):
        self.count = 0
        self.children = {}


class NonNaiveBayes:

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

    # P(x_i = 1 | L) and P(x_i = 1 | S)
    probability_1_in_legitimate = []
    probability_1_in_spam = []

    # Tries for performing lookups of conditional probabilities
    # P(x_i | x_1, ..., x_{i-1}, L) and P(x_i | x_1, ..., x_{i-1}, S)
    legitimate_trie = Node()
    spam_trie = Node()

    def __init__(self, n, legitimate, spam):
        if n <= 0 or legitimate is None or spam is None:
            raise Exception("Invalid arguments")

        self.n = n
        self.probability_1_in_legitimate = [0 for i in range(0, self.n)]
        self.probability_1_in_spam = [0 for i in range(0, self.n)]
        self.likelihood_0 = [0 for i in range(0, self.n)]
        self.likelihood_1 = [0 for i in range(0, self.n)]

        self.legitimate_count = len(legitimate)
        self.spam_count = len(spam)
        self.total_count = self.legitimate_count + self.spam_count

        p_l = self.legitimate_count / float(self.total_count)
        p_s = 1.0 - p_l

        self.threshold = self.misclassification_risk * p_l / p_s

        self.train(legitimate, spam)

        self.search_depths = []

    def train(self, legitimate, spam):
        legitimate_counts = [0 for i in range(0, self.n)]
        spam_counts = [0 for i in range(0, self.n)]

        for message in legitimate:
            for i in range(0, self.n):
                legitimate_counts[i] += message[i]

            self.add_message_to_trie(message, self.legitimate_trie)

        for message in spam:
            for i in range(0, self.n):
                spam_counts[i] += message[i]

            self.add_message_to_trie(message, self.spam_trie)

        for i in range(0, self.n):
            self.probability_1_in_legitimate[i] = legitimate_counts[i] / float(self.legitimate_count)
            self.probability_1_in_spam[i] = spam_counts[i] / float(self.spam_count)

    def add_message_to_trie(self, message, root):
        node = root
        for i in range(0, self.n):
            node.count += 1
            next_node = None
            if message[i] in node.children:
                next_node = node.children[message[i]]
            else:
                next_node = Node()
                node.children[message[i]] = next_node

            node = next_node
        node.count += 1

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
        legitimate_probabilities = self.p_legitimate(vector)
        spam_probabilities = self.p_spam(vector)

        for i in range(0, self.n):
            likelihood *= self.divide(spam_probabilities[i], legitimate_probabilities[i])

        return likelihood > self.threshold

    def p_legitimate(self, vector):
        probability = [0 for i in range(0, self.n)]
        node = self.legitimate_trie
        prev_count = node.count

        for i in range(0, self.n):
            if vector[i] in node.children:
                node = node.children[vector[i]]
                probability[i] = node.count / float(prev_count)
                prev_count = node.count
            else:
                self.search_depths.append(i)
                for j in range(i, self.n):
                    if vector[j] == 1:
                        probability[j] = self.probability_1_in_legitimate[j]
                    else:
                        probability[j] = (1.0 - self.probability_1_in_legitimate[j])
                break

        return probability

    def p_spam(self, vector):
        probability = [0 for i in range(0, self.n)]
        node = self.spam_trie
        prev_count = node.count

        for i in range(0, self.n):
            if vector[i] in node.children:
                node = node.children[vector[i]]
                probability[i] = node.count / float(prev_count)
                prev_count = node.count
            else:
                self.search_depths.append(i)
                for j in range(i, self.n):
                    if vector[j] == 1:
                        probability[j] = self.probability_1_in_spam[j]
                    else:
                        probability[j] = (1.0 - self.probability_1_in_spam[j])
                break

        return probability
