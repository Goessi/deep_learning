import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

# example 1
y_hat = tf.constant(36, name='y_hat')
y = tf.constant(39, name='y')

loss = tf.Variable((y - y_hat)**2,name = 'loss')
init = tf.global_variables_initializer() #初始化loss，因为loss是其他变量的关系式

with tf.Session() as session:
    session.run(init)
    print(session.run(loss))

# steps: 1.创建Tensors还未运行的变量
#        2.基于这些变量的关系式
#        3.初始化Tensors
#        4.创建Session
#        5.运行Session

# example 2
a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a,b)
sess = tf.Session()
print(sess.run(c))

# placeholder
# 设置placeholder时，不用赋值，只用在运行他时利用feed_dict赋值
x = tf.placeholder(tf.int64, name='x')
print(sess.run(2*x, feed_dict={x:3}))
sess.close()

def linear_function():
    X = tf.constant(np.random.randn(3,1), name='X')
    W = tf.constant(np.random.randn(4,3), name='W')
    b = tf.constant(np.random.randn(4,1), name='b')
    Y = tf.constant(np.random.randn(4,1), name='Y')

    sess = tf.Session()
    result = sess.run(tf.add(tf.matmul(W, X), b))
    sess.close()

    return result

# 两个方法去使用session
# method 1
# sess = tf.Session()
# result = sess.run(...,feed_dict = {...})
# sess.close()
# method 2
# with tf.Session() as sess:
#     result = sess.run(...,feed_dict={...})

def sigmoid(z):
    x = tf.placeholder(tf.float32, name='x')
    sigmoid = tf.sigmoid(x)
    with tf.Session() as sess:
        result = sess.run(sigmoid, feed_dict={x:z})
    return result
def cost(logits, labels):
    z = tf.placeholder(tf.float32, name='x')
    y = tf.placeholder(tf.float32, name='y')

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)

    sess = tf.Session()

    cost = sess.run(cost, feed_dict={z:logits, y:labels})

    sess.close()

    return cost

# 'one hot' encoding
def one_hot_matrix(labels, C):
    C = tf.constant(C, name='C')
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)

    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot

def ones(shape):
    ones = tf.ones(shape)
    sess = tf.Session()
    ones = sess.run(ones)
    sess.close()
    return ones
def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=(n_x, None))
    Y = tf.placeholder(tf.float32, shape=(n_y, None))

    return X, Y
def initialize_parameters():
    W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1",[25,1],initializer=tf.zeros_initializer())
    W2 = tf.get_variable('W2',[12,25],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable('b2',[12,1],initializer=tf.zeros_initializer())
    W3 = tf.get_variable('W3', [6,12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable('b3',[6,1], initializer=tf.zeros_initializer())

    parameters = {
        'W1':W1,
        'b1':b1,
        'W2':W2,
        'b2':b2,
        'W3':W3,
        'b3':b3
    }
    return parametres

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1,X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)

    return Z3

def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))

    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,num_epochs=1500,minibatch_size=32,print_cost=True):
    ops.reset_default_graph()
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m/minibatch_size)
            seed += 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict=={X:minibatch_X, Y:minibatch_Y})

                epoch_cost += minibatch_cost/num_minibatches

            if print_cost == True and epoch%5 == 0:
                costs.append(epoch_cost)

        parameters = sess.run(parameters)
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        return parameters

