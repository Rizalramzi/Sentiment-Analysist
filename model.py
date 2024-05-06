import numpy as np

class CustomTfidfVectorizer:
    def __init__(self):
        self.vocab = {}
        self.idf = {}
        self.doc_count = 0

    def fit(self, corpus):
        doc_term_freqs = []
        for doc in corpus:
            term_freqs = {}
            for term in doc.split():
                if term in term_freqs:
                    term_freqs[term] += 1
                else:
                    term_freqs[term] = 1
            doc_term_freqs.append(term_freqs)
            self.doc_count += 1

            for term in term_freqs:
                if term not in self.vocab:
                    self.vocab[term] = len(self.vocab)

        for term in self.vocab:
            doc_freq = sum(1 for doc in doc_term_freqs if term in doc)
            self.idf[term] = np.log(self.doc_count / (doc_freq + 1)) + 1  # Add 1 to avoid division by zero

    def transform(self, corpus):
        rows = []
        cols = []
        values = []

        for i, doc in enumerate(corpus):
            term_freqs = {}
            for term in doc.split():
                if term in term_freqs:
                    term_freqs[term] += 1
                else:
                    term_freqs[term] = 1
            doc_length = sum(term_freqs.values())

            for term, freq in term_freqs.items():
                if term in self.vocab:
                    rows.append(i)
                    cols.append(self.vocab[term])
                    tfidf = (freq / doc_length) * self.idf[term]
                    values.append(tfidf)

        matrix = np.zeros((len(corpus), len(self.vocab)))
        for row, col, value in zip(rows, cols, values):
            matrix[row, col] = value
        return matrix

class CustomKNNClassifier:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))  # Euclidean distance
            nearest_neighbors = np.argsort(distances)[:self.k]  # Indices of k nearest neighbors
            nearest_labels = self.y_train[nearest_neighbors]  # Labels of k nearest neighbors
            unique_labels, counts = np.unique(nearest_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts)]
            y_pred.append(predicted_label)
        return np.array(y_pred)
