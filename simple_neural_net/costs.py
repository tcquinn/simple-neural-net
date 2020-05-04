class Cost:
    def cost(outputs, ground_truth):
        # outputs: (num_outputs, num_examples)
        # cost: Scalar
        raise NotImplementedError('Method must be implemented by child class')

    def d_cost_d_outputs(outputs, ground_truth):
        # outputs: (num_outputs, num_examples)
        # d_cost_d_outputs: (num_outputs, num_examples)
        raise NotImplementedError('Method must be implemented by child class')

class NegativeLogLikelihoodCost(Cost):
    def cost(outputs, ground_truth):
        # outputs: (num_outputs, num_examples)
        # ground_truth: (num_classes, num_examples)
        # cost: Scalar

        # outputs: For this cost function, outputs[i, j] is interpreted as the
        # probability that the jth example is of class i

        # ground_truth: This implementation assumes ground truth is one-hot
        # encoded, with ground_truth[i, j] being 1 when jth example is of class
        # i, 0 otherwise

        num_examples = ground_truth.shape[1]
        return -(1/num_examples)*np.sum(
            np.add(
                np.multiply(
                    ground_truth,
                    np.log(outputs)
                ),
                np.multiply(
                    np.subtract(1.0, ground_truth),
                    np.log(np.subtract(1.0, outputs))
                )
            )
        )

    def d_cost_d_outputs(outputs, ground_truth):
        # outputs: (num_outputs, num_examples)
        # ground_truth: (num_classes, num_examples)
        # d_cost_d_outputs: (num_outputs, num_examples)

        # outputs: For this cost function, outputs[i, j] is interpreted as the
        # probability that the jth example is of class i

        # ground_truth: This implementation assumes ground truth is one-hot
        # encoded, with ground_truth[i, j] being 1 when jth example is of class
        # i, 0 otherwise

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
