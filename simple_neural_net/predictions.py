import numpy as np

class Prediction:
    def predictions(
        self,
        outputs
    ):
        # outputs: (num_outputs, num_examples)
        # predictions: (?, num_examples)
        raise NotImplementedError('Method must be implemented by child class')

class BinaryClassificationPrediction(Prediction):
    def predictions(
        self,
        outputs
    ):
        # outputs: (1, num_examples)
        # predictions: (1, num_examples)
        return np.around(outputs).astype('int')
