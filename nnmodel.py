import tensorflow as tf

# working on the mnist data set
# it contains 60000 training images and 10000 test images
# of numbers from 0 to 9 each are sized 28 by 28

# input -> weight -> hidden layer 1 -> activation funtion
# -> weights -> hidden layer 2 -> activation function
# weights -> output layer

# compare output to labels -> cost function (cross entropy)
# optimization function -> minimize the cost with Adam
# Optimizer, SGD, AdaGrad etc
# (backpropagation)

# feedforward + backpropagation = epoch

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_classes = 10

# we will have 3 hidden layers
n_nodes_hiddenlayer1 = 500
n_nodes_hiddenlayer2 = 500
n_nodes_hiddenlayer3 = 500

#Batch size defines number of samples that going to be propagated through the network.
batch_size = 100

# images do not have to be in the shape of an 2d array:
x = tf.placeholder('float', [None, 28*28])
y = tf.placeholder('float')


def neural_network_model(data):
    hidden_layer1 = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hiddenlayer1])),
                     'biases': tf.Variable(tf.random_normal([n_nodes_hiddenlayer1]))}

    hidden_layer2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hiddenlayer1, n_nodes_hiddenlayer2])),
                     'biases': tf.Variable(tf.random_normal([n_nodes_hiddenlayer2]))}

    hidden_layer3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hiddenlayer2, n_nodes_hiddenlayer3])),
                     'biases': tf.Variable(tf.random_normal([n_nodes_hiddenlayer3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hiddenlayer3, n_classes])),
                     'biases': tf.Variable(tf.random_normal([n_classes]))}

    layer1 = tf.add(tf.matmul(data, hidden_layer1["weights"]), hidden_layer1["biases"])
    # apply the activation function
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.add(tf.matmul(layer1, hidden_layer2["weights"]), hidden_layer2["biases"])
    # apply the activation function
    layer2 = tf.nn.relu(layer2)

    layer3 = tf.add(tf.matmul(layer2, hidden_layer3["weights"]), hidden_layer3["biases"])
    # apply the activation function
    layer3 = tf.nn.relu(layer3)

    output = tf.add(tf.matmul(layer3, output_layer["weights"]), output_layer["biases"])

    return output


# runs data through our model
def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    # minimize the cost (minimize the difference between true labels and
    # out prediction
    #learning rate = 0.001 by default
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # number of cycles of feedforward + backprop
    num_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Training step:
        for epoch in range(num_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                # chunks through the data set for you:
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print("Epoch", epoch, "completed out of", num_epochs, "loss:", epoch_loss)

        # Run on the test data
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)