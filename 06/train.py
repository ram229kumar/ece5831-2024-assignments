import numpy as np
import pickle
from two_layer_net_with_back_prop import TwoLayerNetWithBackProp
from mnist_data import MnistData

input_size = 784
hidden_size = 100
output_size = 10
iterationSize = 10000
batchSize = 16
learningRate = 0.01
mnist = MnistData()

(x_train,y_train),(x_test,y_test) = mnist.load()
train_size = x_train.shape[0]
iter_per_epoch = max(train_size // batchSize, 1)

network = TwoLayerNetWithBackProp(input_size, hidden_size, output_size)
y_hat = network.predict(x_test[0:100])

train_losses = []
train_accs = []
test_accs = []

for i in range(iterationSize):
    batch_mask = np.random.choice(train_size, batchSize)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]

    grads = network.gradient(x_batch, y_batch)

    for key in ('w1', 'b1', 'w2', 'b2'):
        network.params[key] -= learningRate*grads[key]

    ## this is for plotting losses over time
    train_losses.append(network.loss(x_batch, y_batch))

    if i%iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, y_train)
        train_accs.append(train_acc)
        test_acc = network.accuracy(x_test, y_test)
        test_accs.append(test_acc)
        print(f'train acc, test_acc : {train_acc}, {test_acc}')


my_weight_pkl_file = 'ippili_mnist_model.pkl'
with open(f'{my_weight_pkl_file}', 'wb') as f:
    print(f'Pickle: {my_weight_pkl_file} is being created.')
    pickle.dump(network.params, f)
    print('Done.') 