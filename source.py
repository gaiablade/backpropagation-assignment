import csv
import math
import numpy as np
import random as rn

"""
            ### CS460G Programming Assignment 3 ###
    ## Binary Classification Using Multilayer Perceptrons ##

Training with backpropagation. Will report overall accuracy of your model.
* Your network can have one or two output nodes.
* Must have at least one hidden layer.
* Can use either sigmoid or rectifier function.
* Don't forget the bias term!

Sigmoid activation function: g(in[i]) = 1 / (1 + e^-in[i])
"""

rn.seed(1251999)

def open_csv(filename: str):
    a = []
    with open(filename) as file:
        reader = csv.reader(file)
        for row in reader:
            a.append([int(val) for val in row])
    return a

def print_number(arr: list):
    for i in range(28):
        s = ''
        for j in range(28):
            s += format(arr[i * 28 + j + 1], '03d') + ' '
        print(s)

def sigmoid(ini: float):
    return 1.0 / (1.0 + math.exp(-ini))

activation_func = np.vectorize(sigmoid)

def normalize(data: list):
    samples = data.copy()
    for sample in samples:
        for val in range(1, len(sample)):
            sample[val] = float(sample[val]) / 255.0
    return samples

def predict(x, w_i2h, w_h2o, bias_h, bias_o):
    in_h = np.matmul(w_i2h.T, x) + bias_h
    a_h  = activation_func(in_h)

    in_o = np.matmul(w_h2o.T, a_h) + bias_o
    a_o  = activation_func(in_o)
    return a_o[0][0]

train = open_csv("data/mnist_train_0_1.csv")
train = normalize(train)
test  = open_csv("data/mnist_test_0_1.csv")
test  = normalize(test)

# There are 784 inputs
n_input  = 784
n_hidden = 10
n_output = 1
alp      = 0.5

w_i2h       = np.array([[rn.random() / 1000.0 for i in range(n_hidden)] for j in range(n_input)]) # [[Wx1h1, Wx1h2, ...], [Wx2h1, Wx2h2, ...], ...]
w_h2o       = np.array([[rn.random() / 1000.0] for i in range(n_hidden * n_output)])                # [Wh1o1, Wh2o1, Wh3o1, ...]
bias_h      = np.array([[rn.random() / 1000.0] for i in range(n_hidden)])
bias_o      = np.array([[rn.random() / 1000.0] for i in range(n_output)])

delta_hidden = None
delta_output = None

for iteration in range(500):
    if iteration % 100 == 0:
        print(f'Iteration #{iteration}')
        print(f'w_h2o: {w_h2o}')
    # Get random number of samples
    # FOR NOW: we'll just use the first 50
    samples = rn.sample(train, 300)
    for sample in samples:
        ### Calculate output
        x    = np.array([sample[1:]]).T

        in_h = np.matmul(w_i2h.T, x) + bias_h
        a_h  = activation_func(in_h)

        in_o = np.matmul(w_h2o.T, a_h) + bias_o
        a_o  = activation_func(in_o)

        ### Calculate and backprop deltas
        # output:
        ground_truth = sample[0]
        delta_output = np.array([[(ground_truth - a_o[0][0]) * (a_o[0][0]) * (1-a_o[0][0])]]) # Err - g'(in)

        # hidden:
        delta_hidden = np.matmul(w_h2o, delta_output) * (a_h * (1 - a_h)) # w*delta * g'(in)

        ### Update weights
        # output
        w_h2o = w_h2o + (alp*a_h*delta_output)
        bias_o = bias_o + alp*delta_output

        # hidden
        w_i2h = w_i2h + (alp*np.matmul(x,delta_hidden.T))
        bias_h = bias_h + alp*delta_hidden

print(f'final weights:')
print(f'{w_h2o}')

num_test_samples = len(test)
num_correct_predictions = 0
for sample in test:
    x = sample[1:]
    y = sample[0]
    yp = predict(x, w_i2h, w_h2o, bias_h, bias_o)
    o = 0 if yp < 0.5 else 1
    print(f'Expected: {y}, Predicted: {o} {"CORRECT" if y == o else "WRONG"}')
    if o == y: num_correct_predictions += 1

print(f'num_correct: {num_correct_predictions}')
print(f'accuracy: {(float(num_correct_predictions) / float(num_test_samples)) * 100}%')