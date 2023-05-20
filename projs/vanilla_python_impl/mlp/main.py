import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

from model import Linear, ReLU, Softmax, CrossEntropy, Model
from data_process import load_data, one_hot, data_iter


def get_args():
    parser = argparse.ArgumentParser(description='MLP for MNIST with Vanilla Python')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--num-hidden-layers', type=int, default=2, help='the number of hidden layers')
    parser.add_argument('--num-classes', type=int, default=10, help='the number of classes')
    parser.add_argument('--num-epoch', type=int, default=100, help='the number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--download', type=bool, default=False, help='whether download dataset')
    parser.add_argument('--process-mnist', type=bool, default=False, help='whether process mnist')
    parser.add_argument(
        '--layer-size',
        type=int,
        nargs='+',
        default=(784, 200, 100, 10),
        help='size of layers of MLP, e.g. (784, 200, 100, 10) means 784 -> 200 -> 100 -> 10',
    )
    parser.add_argument('--dataset-root', default='mlp_mnist/', type=str, help='dataset root directory')
    parser.add_argument('--save-root', default='mlp_mnist/', type=str, help='image save root directory')

    return parser.parse_args()


def build_mlp(args):
    """Return a MLP according to the parameters in args."""
    model_list = []

    # build hidden layers with linear and activation
    for i in range(args.num_hidden_layers):
        model_list.append(Linear(args.layer_size[i], args.layer_size[i + 1]))
        model_list.append(ReLU())
        print(f'Linear({args.layer_size[i]}, {args.layer_size[i + 1]}) -> Relu()', end=' -> ')

    # build output layer and softmax
    model_list.append(Linear(args.layer_size[-2], args.layer_size[-1]))
    model_list.append(Softmax())
    print(f'Linear({args.layer_size[-2]}, {args.layer_size[-1]}) -> Softmax() -> CrossEntropy()', end='\n\n')

    return Model(model_list, CrossEntropy())


def draw(args, running_loss, testing_acc):
    """Draw curve."""
    x = np.arange(len(running_loss)) + 1

    plt.subplot(121)
    plt.plot(x, running_loss, 'bo--')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(122)
    plt.plot(x, testing_acc, 'ro--')
    plt.title('Testing Acc Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')

    plt.suptitle('MLP MNIST Curve')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_root, 'curve.png'))


if __name__ == '__main__':
    args = get_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    mlp = build_mlp(args)

    x_train, t_train, x_test, t_test = load_data(args.dataset_root,
                                                 download=args.download,
                                                 process_mnist=args.process_mnist)
    t_train = one_hot(args.num_classes, t_train)

    # record the running loss and testing acc for drawing curve chart
    rec_running_loss = []
    rec_testing_acc = []

    for epoch in range(args.num_epoch):
        # training
        running_loss = 0.
        num_inputs = 0

        print(f'[Epoch {epoch + 1}/{args.num_epoch}] training')

        for i, mini_batch in enumerate(data_iter(x_train, t_train, args.batch_size, shuffle=True)):
            inputs, targets = mini_batch
            num_inputs += inputs.shape[0]

            outputs = mlp.forward(inputs)
            loss = mlp.compute_loss(outputs, targets)
            running_loss += loss.sum()

            mlp.backward()

            for layer in mlp.layers:
                # only Linear layer need to be updated in MLP
                if isinstance(layer, Linear):
                    layer.weights -= args.lr * layer.grad_w
                    layer.biases -= args.lr * layer.grad_b

        print(f'[Epoch {epoch + 1}/{args.num_epoch}] total_mean_loss = {running_loss / num_inputs}')
        rec_running_loss.append(running_loss / num_inputs)

        # testing
        num_inputs = 0
        num_correct = 0

        print(f'[Epoch {epoch + 1}/{args.num_epoch}] testing')

        for i, mini_batch in enumerate(data_iter(x_test, t_test, args.batch_size, shuffle=False)):
            inputs, targets = mini_batch
            num_inputs += inputs.shape[0]

            outputs = mlp.forward(inputs)
            num_correct += np.sum(np.argmax(outputs, axis=1) == targets)

        print(f'[Epoch {epoch + 1}/{args.num_epoch}] test_accuracy = {num_correct / num_inputs}')
        rec_testing_acc.append(num_correct / num_inputs)

    draw(args, rec_running_loss, rec_testing_acc)
