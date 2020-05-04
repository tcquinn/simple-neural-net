import numpy as np

class Cost:
    def cost(outputs, ground_truth):
        # outputs: (num_outputs, num_examples)
        # cost: Scalar
        raise NotImplementedError('Method must be implemented by child class')

    def d_cost_d_outputs(outputs, ground_truth):
        # outputs: (num_outputs, num_examples)
        # d_cost_d_outputs: (num_outputs, num_examples)
        raise NotImplementedError('Method must be implemented by child class')

class CrossEntropyBinaryClassification(Cost):
    def cost(
        self,
        outputs,
        ground_truth
    ):
        # outputs: (1, num_examples)
        # ground_truth: (1, num_examples)
        # cost: Scalar

        # outputs: For this cost function, outputs[0, i] is interpreted as the
        # probability that the ith example is of class 1 (as opposed to class 0)

        # ground_truth: For this cost function, ground_truth[0, i] is either 0
        # (first class) or 1 (second class)

        num_examples = ground_truth.shape[1]
        return np.squeeze(-(1/num_examples)*np.sum(
            np.add(
                np.multiply(
                    ground_truth,
                    np.log(outputs)
                ),
                np.multiply(
                    np.subtract(1.0, ground_truth),
                    np.log(np.subtract(1.0, outputs))
                )
            ),
            axis=1
        ))

    def d_cost_d_outputs(
        self,
        outputs,
        ground_truth
    ):
        # outputs: (num_outputs, num_examples)
        # ground_truth: (num_classes, num_examples)
        # d_cost_d_outputs: (num_outputs, num_examples)

        # outputs: For this cost function, outputs[0, i] is interpreted as the
        # probability that the ith example is of class 1 (as opposed to class 0)

        # ground_truth: For this cost function, ground_truth[0, i] is either 0
        # (first class) or 1 (second class)

        num_examples = ground_truth.shape[1]
        return -(1/num_examples)*np.divide(
            np.subtract(
                ground_truth,
                outputs
            ),
            np.multiply(
                outputs,
                np.subtract(1, outputs)
            )
        )
