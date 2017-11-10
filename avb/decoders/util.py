import tensorflow as tf
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

