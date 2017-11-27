import tensorflow as tf
import numpy as np
import cv2
def get_alpha(r,N,index):

    if isinstance(r , list)  :
        alpah = []
        for i in range(len(index)):
        #for i in range(index):

            alpah.append(tf.linspace(r[i][0], r[i][1], N))
    return alpah


def get_zero_matrx(z_dim, N):
    zero = tf.zeros(N)
    zeros = [zero for i in range(z_dim)]
    return zeros


def get_inc(zeros, index, alpah, z_dim):
    new_array = [zeros[i] if i != index else alpah for i in range(z_dim)]
    return tf.transpose(tf.stack(new_array), (1, 0))


def get_all_inc(index, zeros, alpah, z_dim):
    all_inc = []
    for i in range(len(index)):
        all_inc.append(get_inc(zeros, index[i], alpah[i], z_dim))
    return tf.concat(all_inc, axis = 0)

def get_add_result(z,all_inc,index):
    #z[index[0]]=0
    return z + all_inc


def get_interplate_Z(z, r, index, N, z_dim):
    alpah = get_alpha(r, N,index)
    zeros = get_zero_matrx(z_dim, N)
    all_inc = get_all_inc(index, zeros, alpah, z_dim)
    z_interplat = get_add_result(z, all_inc,index)
    return z_interplat


def change(x, index = 0, range_ = (0, 1), n = 5):
    n = int(n)
    # print(n)
    x = np.repeat(x, n, 0)
    # print(x.shape)
    x[:, index] = x[:, index] + np.linspace(range_[0], range_[1], n)
    return x


def multi_chane(x, index = [0, 1], range_ = [(0, 1), (2, 3)], n = 5):
    all_c = []
    for i in range(len(index)):
        all_c.append(change(x, index[i], range_[i], n))
    all_c = np.concatenate(all_c)
    return all_c


def cvt_imgs(decode_img, N = 10, name = 'test.jpg'):
    n = N
    size = decode_img.shape[0]

    big_img = np.ones([int(64 * size / n), int(64 * n), 3], np.float32)

    o_shape = [64, 64]
    for i in range(int(size / n)):
        for j in range(N):
            addimg = decode_img[i * n + j][:]
            big_img[i * o_shape[0]:(i + 1) * o_shape[0], j * o_shape[1]:(j + 1) * o_shape[1]] = addimg
    # big_img = (big_img + 1.) * 255. / 2. #这里可能需要修改
    big_img = (big_img) * 255.
    big_img_rgb = cv2.cvtColor(big_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(name, big_img_rgb)


