from Convolutional_NN_module import ConvolutionalNN
from FFNN_module import FeedForwarnNN
from RandomForest_module import RandomForest

class MnistClassifier:
    def __init__(self, algorithm: str):
        if algorithm == 'cnn':
            self.model = ConvolutionalNN()
        elif algorithm == 'nn':
            self.model = FeedForwarnNN()
        elif algorithm == 'rf':
            self.model = RandomForest()
        else:
            raise ValueError("Invalid algorithm. Choose from 'cnn', 'nn', 'rf'.")

    def train(self, X_train, y_train):

        return self.model.train(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

