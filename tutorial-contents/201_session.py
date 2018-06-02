"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.8.0
"""
import tensorflow as tf


def main(argv):
    m1 = tf.constant([[2, 2]])
    m2 = tf.constant([[3],
                      [3]])

    # m2的另外一种写法，两行一列，填充数字 3 的矩阵
    m3 = tf.constant(3, shape=[2, 1])

    dot_operation = tf.matmul(m1, m2)
    dot_operation_ = tf.matmul(m1, m3)

    print(dot_operation)  # wrong! no result

    # method1 Using the `close()` method.
    sess = tf.Session()
    result = sess.run(dot_operation)
    print(result)
    sess.close()

    # method2 Using the context manager.
    with tf.Session() as sess:
        # result_ = sess.run(dot_operation)
        # print(result_)
        result_ = sess.run(dot_operation_)
        print(result_)

    # tf.constant
    # Constant 1-D Tensor populated with value list.
    # [1 2 3 4 5 6 7]
    tensor = tf.constant([1, 2, 3, 4, 5, 6, 7])

    print(tensor)

    # Constant 2-D tensor populated with scalar value -1.
    # [[-1. - 1. - 1.]
    # [-1. - 1. - 1.]]
    tensor = tf.constant(-1.0, shape=[2, 3])

    print(tensor)

    # Lists available devices in this session.
    devices = sess.list_devices()
    for d in devices:
        print(d.name)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)