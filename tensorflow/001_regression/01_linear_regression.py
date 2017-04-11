import numpy as np
import tensorflow as tf

x_data = np.random.rand(100, 1)
y_data = x_data * 7 + 12

x_ = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_ = tf.placeholder(shape=[None, 1], dtype=tf.float32)

W = tf.Variable(tf.random_normal([1, 1]))
b = tf.Variable(tf.random_normal([1, 1]))

pred = tf.add(tf.matmul(x_, W), b)
## (y_-pred)平方  然后求平均值
loss = tf.reduce_mean(tf.square(y_ - pred))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss=loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train_step, feed_dict={x_: x_data, y_: y_data})
        loss1, W1, b1 = sess.run([loss, W, b], feed_dict={x_: x_data, y_: y_data})
        print(i, loss1, W1, b1)
