import numpy as np
import scipy.io
import tensorflow as tf
mat = scipy.io.loadmat('firingrate.mat')['firingrate']
print mat.shape
x_train = np.zeros((95, 160 * 8))
y_train = np.zeros((160 * 8, 8))
x_test = np.zeros((95, 22 * 8))
y_test = np.zeros((22 * 8, 8))
train_temp = 0
test_temp = 0
print x_train.shape
for i in range(8):
	for j in range(182):
		if j < 160:
			x_train[:, train_temp] = mat[:, j, i]
			y_train[train_temp, i] = 1
			train_temp += 1
		else:
			x_test[:, test_temp] = mat[:, j, i]
			y_test[test_temp, i] = 1
			test_temp += 1
x_train = x_train.transpose()
print x_train[1279, :]
x_test = x_test.transpose()
x = tf.placeholder(tf.float32, [None, 95])
W = tf.Variable(tf.zeros([95, 8]))
b = tf.Variable(tf.zeros([8]))
y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 8])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.initialize_all_variables().run()
for i in range(10000):
  	sess.run(train_step, feed_dict={x: x_train, y_: y_train})
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))