import functools
from typing import Callable, Tuple, List

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

import auto_diff as ad
import torch
from torchvision import datasets, transforms

max_len = 28


# ==========================
# p1: Implement Linear Layer
# ==========================
def linear(x: ad.Node, W: ad.Node, b: ad.Node) -> ad.Node:
    """Linear transformation: output = x @ W + b

    Parameters
    ----------
    x: ad.Node
        Input tensor of any shape, last dimension will be transformed
    W: ad.Node
        Weight matrix for linear transformation
    b: ad.Node
        Bias vector

    Returns
    -------
    output: ad.Node
        Transformed tensor
    """
    return ad.add(ad.matmul(x, W), b)


# =============================
# p2: Implement Attention Layer
# =============================
def attention(x: ad.Node, W_Q: ad.Node, W_K: ad.Node, W_V: ad.Node, model_dim: int) -> ad.Node:
    """Construct the computational graph for a single-head self-attention layer."""
    # perform matmul to get q, k and v
    q = ad.matmul(x, W_Q)
    k = ad.matmul(x, W_K)
    v = ad.matmul(x, W_V)

    # compute attention scores and normalize
    score = ad.matmul(q, ad.transpose(k, 1, 2))
    norm_score = ad.div_by_const(score, (model_dim ** 0.5))

    # softmax the scores to get probabilities
    attn_weights = ad.softmax(norm_score, dim=-1)
    # return the weighted sum of the values
    return ad.matmul(attn_weights, v)


def transformer(X: ad.Node, nodes: List[ad.Node], model_dim: int, num_classes: int, eps: float) -> ad.Node:
    """Construct the computational graph for a single transformer layer with sequence classification.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, seq_length, input_dim), denoting the input data.
    nodes: List[ad.Node]
        Nodes [W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2] for the transformer.
    model_dim: int
        Dimension of the model (hidden size).
    eps: float
        Epsilon parameter (for potential use in normalization).

    Returns
    -------
    output: ad.Node
        The output of the transformer layer, shape (batch_size, seq_length, num_classes).
    """
    W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2 = nodes

    # perform attention:
    att_output = attention(X, W_Q, W_K, W_V, model_dim)
    att_output = ad.matmul(att_output, W_O)
    norm_output = ad.layernorm(att_output, [model_dim], eps=eps)

    # perform feed-forward:
    ffn_hidden = ad.relu(linear(norm_output, W_1, b_1))
    ffn_hidden = linear(ffn_hidden, W_2, b_2)
    norm_output = ad.layernorm(ffn_hidden, [num_classes], eps=eps)

    return norm_output


def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """Construct the computational graph of average softmax loss over
    a batch of logits.

    Parameters
    ----------
    Z: ad.Node
        A node in of shape (batch_size, num_classes) or (batch_size, seq_length, num_classes),
        containing the logits for the batch of instances.

    y_one_hot: ad.Node
        A node in of shape (batch_size, num_classes) or (batch_size, seq_length, num_classes),
        containing the one-hot encoding of the ground truth label for the batch of instances.

    batch_size: int
        The size of the mini-batch.

    Returns
    -------
    loss: ad.Node
        Average softmax loss over the batch.
        When evaluating, it should be a zero-rank array (i.e., shape is `()`).

    Note
    ----
    1. In this homework, you do not have to implement a numerically
    stable version of softmax loss.
    2. You may find that in other machine learning frameworks, the
    softmax loss function usually does not take the batch size as input.
    Try to think about why our softmax loss may need the batch size.
    """
    # compute log probs and then cross entropy
    log_probs = ad.log(ad.softmax(Z, dim=-1))
    cross_entropy = ad.mul(y_one_hot, log_probs)

    total_loss = ad.sum_op(cross_entropy, dim=2, keepdim=True)  # sum over num_classes
    total_loss = ad.sum_op(total_loss, dim=1, keepdim=True)     # sum over seq_length
    total_loss = ad.sum_op(total_loss, dim=0, keepdim=True)     # sum over batch_size

    # average over batch size
    avg_loss = ad.div_by_const(total_loss, float(batch_size))
    return ad.mul_by_const(avg_loss, -1.0)  # negate the loss


def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: List[torch.Tensor],
    batch_size: int,
    lr: float,
) -> List[torch.Tensor]:
    """Run an epoch of SGD for the logistic regression model
    on training data with regard to the given mini-batch size
    and learning rate.

    Parameters
    ----------
    f_run_model: Callable
        The function to run the forward and backward computation
        at the same time for logistic regression model.
        It takes the training data, training label, model weight
        and bias as inputs, and returns the logits, loss value,
        weight gradient and bias gradient in order.
        Please check `f_run_model` in the `train_model` function below.

    X: torch.Tensor
        The training data in shape (num_examples, in_features).

    y: torch.Tensor
        The training labels in shape (num_examples,).

    model_weights: List[torch.Tensor]
        The model weights in the model.

    batch_size: int
        The mini-batch size.

    lr: float
        The learning rate.

    Returns
    -------
    model_weights: List[torch.Tensor]
        The model weights after update in this epoch.

    b_updated: torch.Tensor
        The model weight after update in this epoch.

    loss: torch.Tensor
        The average training loss of this epoch.
    """
    num_examples = X.shape[0]
    num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
    total_loss = 0.0

    for i in range(num_batches):
        # Get the mini-batch data
        start_idx = i * batch_size
        if start_idx + batch_size > num_examples:continue
        end_idx = min(start_idx + batch_size, num_examples)
        X_batch = X[start_idx:end_idx, :max_len]
        y_batch = y[start_idx:end_idx]

        # Compute forward and backward passes
        logits, loss_val, *grads = f_run_model(X_batch, y_batch, model_weights)

        model_weights[0] = model_weights[0] - lr * grads[0].sum(dim=0)
        model_weights[1] = model_weights[1] - lr * grads[1].sum(dim=0)
        model_weights[2] = model_weights[2] - lr * grads[2].sum(dim=0)
        model_weights[3] = model_weights[3] - lr * grads[3].sum(dim=0)
        model_weights[4] = model_weights[4] - lr * grads[4].sum(dim=0)
        model_weights[5] = model_weights[5] - lr * grads[5].sum(dim=0)
        model_weights[6] = model_weights[6] - lr * grads[6].sum(dim=0)
        model_weights[7] = model_weights[7] - lr * grads[7].sum(dim=0)

        # Accumulate the loss
        loss_item = loss_val.item() if hasattr(loss_val, 'item') else float(loss_val)
        total_loss += loss_item * (end_idx - start_idx)

    # Compute the average loss
    average_loss = total_loss / num_examples
    print('Avg_loss:', average_loss)

    return model_weights, average_loss


def train_model():
    """Train a logistic regression model with handwritten digit dataset.

    Note
    ----
    Your implementation should NOT make changes to this function.
    """
    # Set up model params
    input_dim = 28        # Each row of the MNIST image
    seq_length = max_len  # Number of rows in the MNIST image
    num_classes = 10      # Number of classes to classify
    model_dim = 128       # model hidden dimension
    eps = 1e-5            # 

    # Set up tain params
    num_epochs = 20
    batch_size = 50
    lr = 0.02

    # Create variable nodes for inputs
    X = ad.Variable(name="X")
    W_Q = ad.Variable(name="W_Q")
    W_K = ad.Variable(name="W_K")
    W_V = ad.Variable(name="W_V")
    W_O = ad.Variable(name="W_O")
    W_1 = ad.Variable(name="W_1")
    W_2 = ad.Variable(name="W_2")
    b_1 = ad.Variable(name="b_1")
    b_2 = ad.Variable(name="b_2")
    y = ad.Variable(name="y")

    nodes = [W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2]

    y_predict: ad.Node = transformer(X, nodes, model_dim, num_classes, eps)
    loss: ad.Node = softmax_loss(y_predict, y, batch_size)

    # Compute gradients with respect to all weight parameters
    grads: List[ad.Node] = ad.gradients(loss, nodes)
    evaluator = ad.Evaluator([y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])

    # Prepare the MNIST dataset for training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # convert the train dataset and test dataset to NumPy arrays
    X_train = train_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
    y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
    y_test = test_dataset.targets.numpy()

    # Initialize the OneHotEncoder, then fit and transform y_train and y_test
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))

    num_classes = 10

    # Initialize model weights.
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)
    W_Q_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_K_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_V_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_O_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_1_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_2_val = np.random.uniform(-stdv, stdv, (model_dim, num_classes))
    b_1_val = np.random.uniform(-stdv, stdv, (model_dim,))
    b_2_val = np.random.uniform(-stdv, stdv, (num_classes,))

    def f_run_model(X_batch, y_batch, model_weights):
        """The function to compute the forward and backward graph.
        It returns the logits, loss, and gradients for model weights.
        """
        # expand (batch_size, num_classes) to (batch_size, seq_length, num_classes)
        y_batch_expanded = np.repeat(y_batch[:, np.newaxis, :], seq_length, axis=1)

        result = evaluator.run(
            input_values={
                X: torch.DoubleTensor(X_batch),
                y: torch.DoubleTensor(y_batch_expanded),
                W_Q: torch.DoubleTensor(model_weights[0]),
                W_K: torch.DoubleTensor(model_weights[1]),
                W_V: torch.DoubleTensor(model_weights[2]),
                W_O: torch.DoubleTensor(model_weights[3]),
                W_1: torch.DoubleTensor(model_weights[4]),
                W_2: torch.DoubleTensor(model_weights[5]),
                b_1: torch.DoubleTensor(model_weights[6]),
                b_2: torch.DoubleTensor(model_weights[7]),
            }
        )
        return result

    def f_eval_model(X_val, model_weights: List[torch.Tensor]):
        """The function to compute the forward graph only and returns the prediction."""
        num_examples = X_val.shape[0]
        num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
        all_logits = []
        for i in range(num_batches):
            # Get the mini-batch data
            start_idx = i * batch_size
            if start_idx + batch_size > num_examples:continue
            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx, :max_len]
            logits = test_evaluator.run({
                X: torch.DoubleTensor(X_batch),
                W_Q: torch.DoubleTensor(model_weights[0]),
                W_K: torch.DoubleTensor(model_weights[1]),
                W_V: torch.DoubleTensor(model_weights[2]),
                W_O: torch.DoubleTensor(model_weights[3]),
                W_1: torch.DoubleTensor(model_weights[4]),
                W_2: torch.DoubleTensor(model_weights[5]),
                b_1: torch.DoubleTensor(model_weights[6]),
                b_2: torch.DoubleTensor(model_weights[7]),
            })
            # logits shape is (batch_size, seq_length, num_classes), average over seq_length
            logits_arr = logits[0]  # (batch_size, seq_length, num_classes)
            logits_avg = logits_arr.mean(axis=1)  # (batch_size, num_classes)
            all_logits.append(logits_avg)
        # Concatenate all logits and return the predicted classes
        concatenated_logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions

    # prepare data and initialize model weights:
    X_train, X_test, y_train, y_test= torch.tensor(X_train), torch.tensor(X_test), torch.DoubleTensor(y_train), torch.DoubleTensor(y_test)
    model_weights: List[torch.Tensor] = [
        torch.DoubleTensor(W_Q_val),
        torch.DoubleTensor(W_K_val),
        torch.DoubleTensor(W_V_val),
        torch.DoubleTensor(W_O_val),
        torch.DoubleTensor(W_1_val),
        torch.DoubleTensor(W_2_val),
        torch.DoubleTensor(b_1_val),
        torch.DoubleTensor(b_2_val),
    ]

    # Train the model using SGD
    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        model_weights, loss_val = sgd_epoch(
            f_run_model, X_train, y_train, model_weights, batch_size, lr
        )

        # Evaluate the model on the test data.
        predict_label = f_eval_model(X_test, model_weights)
        print(
            f"Epoch {epoch}: test accuracy = {np.mean(predict_label== y_test.numpy())}, "
            f"loss = {loss_val}"
        )

    # Return the final test accuracy.
    predict_label = f_eval_model(X_test, model_weights)
    return np.mean(predict_label == y_test.numpy())


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")
