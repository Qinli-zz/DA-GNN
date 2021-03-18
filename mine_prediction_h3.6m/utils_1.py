import numpy as np

import copy

import torch

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error



def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv1d') != -1:

        m.weight.data.normal_(0.0, 0.02)

        if m.bias is not None:

            m.bias.data.fill_(0)

    elif classname.find('Conv2d') != -1:

        m.weight.data.normal_(0.0, 0.02)

        if m.bias is not None:

            m.bias.data.fill_(0)

    elif classname.find('BatchNorm') != -1:

        m.weight.data.normal_(1.0, 0.02)

        m.bias.data.fill_(0)



def read_txt_as_data(filename):

    returnArray = []

    lines = open(filename).readlines()

    for line in lines:

        line = line.strip().split(',')

        if len(line) > 0:

            returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)

    return returnArray



def load_data(data_path, subjects, actions):

    nactions = len(actions)

    sampled_data_set, complete_data = {}, []

    for subj in subjects:

        for action_idx in np.arange(nactions):

            action = actions[action_idx]

            for subact in [1, 2]:

                print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))

                filename = '{0}/S{1}/{2}_{3}.txt'.format(data_path, subj, action, subact)

                action_sequence = read_txt_as_data(filename)

                t, d = action_sequence.shape

                even_indices = range(0, t, 2)

                sampled_data_set[(subj, action, subact, 'even')] = action_sequence[even_indices, :]

                if len(complete_data) == 0:

                    complete_data = copy.deepcopy(action_sequence)

                else:

                    complete_data = np.append(complete_data, action_sequence, axis=0)

    return sampled_data_set, complete_data



def train_sample(data_set, batch_size, source_seq_len, target_seq_len, input_size):



    all_keys = list(data_set.keys())

    chosen_keys_idx = np.random.choice(len(all_keys), batch_size)

    total_seq_len = source_seq_len + target_seq_len



    encoder_inputs  = np.zeros((batch_size, source_seq_len, input_size), dtype=np.float32)

    decoder_inputs = np.zeros((batch_size, 1, input_size), dtype=np.float32)

    decoder_outputs = np.zeros((batch_size, target_seq_len, input_size), dtype=np.float32)



    for i in range(batch_size):

        the_key = all_keys[chosen_keys_idx[i]]

        t, d = data_set[the_key].shape

        idx = np.random.randint(16, t-total_seq_len)

        data_sel = data_set[the_key][idx:idx+total_seq_len,:]



        encoder_inputs[i,:,:]  = data_sel[0:source_seq_len,:]

        decoder_inputs[i,:,:]  = data_sel[source_seq_len-1:source_seq_len,:]

        decoder_outputs[i,:,:] = data_sel[source_seq_len:,:]



    # rs = int(np.random.uniform(low=0, high=4))
    #
    # downsample_idx = np.array([int(i)+rs for i in [np.floor(j*4) for j in range(12)]])



    return encoder_inputs, decoder_inputs, decoder_outputs



def srnn_sample(data_set, action, source_seq_len, target_seq_len, input_size):

    frames = {}

    frames[action] = find_indices_srnn(data_set, action)

    batch_size, subject = 8, 5

    seeds = [(action, (i%2)+1, frames[action][i]) for i in range(batch_size)]



    encoder_inputs = np.zeros((batch_size, source_seq_len, input_size), dtype=np.float32)

    decoder_inputs = np.zeros((batch_size, 1, input_size), dtype=np.float32)

    decoder_outputs = np.zeros((batch_size, target_seq_len, input_size), dtype=np.float32)

    for i in range(batch_size):

        _, subsequence, idx = seeds[i]

        idx = idx+source_seq_len

        data_sel = data_set[(subject, action, subsequence, 'even')]

        data_sel = data_sel[(idx-source_seq_len):(idx+target_seq_len), :]

        encoder_inputs[i, :, :] = data_sel[0:source_seq_len, :]

        decoder_inputs[i, :, :] = data_sel[source_seq_len-1:source_seq_len, :]

        decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]



    rs = int(np.random.uniform(low=0, high=4))

    downsample_idx = np.array([int(i)+rs for i in [np.floor(j*4) for j in range(12)]])



    return encoder_inputs, decoder_inputs, decoder_outputs



def find_indices_srnn(data_set, action):

    seed = 1234567890

    rng = np.random.RandomState(seed)

    subject = 5

    T1 = data_set[(subject, action, 1, 'even')].shape[0]

    T2 = data_set[(subject, action, 2, 'even')].shape[0]

    prefix, suffix = 50, 100



    idx = []

    idx.append(rng.randint(16, T1-prefix-suffix))

    idx.append(rng.randint(16, T2-prefix-suffix))

    idx.append(rng.randint(16, T1-prefix-suffix))

    idx.append(rng.randint(16, T2-prefix-suffix))

    idx.append(rng.randint(16, T1-prefix-suffix))

    idx.append(rng.randint(16, T2-prefix-suffix))

    idx.append(rng.randint(16, T1-prefix-suffix))

    idx.append(rng.randint(16, T2-prefix-suffix))

    return idx



def rotmat2euler(R):

    if R[0,2]==1 or R[0,2]==-1:

        e3 = 0

        dlta = np.arctan2(R[0,1], R[0,2])

        if R[0,2]==-1:

            e2 = np.pi/2

            e1 = e3+dlta

        else:

            e2 = -1*np.pi/2

            e1 = -1*e3+dlta

    else:

        e2 = -1*np.arcsin(R[0,2])

        e1 = np.arctan2(R[1,2]/np.cos(e2), R[2,2]/np.cos(e2))

        e3 = np.arctan2(R[0,1]/np.cos(e2), R[0,0]/np.cos(e2))

    eul = np.array([e1, e2, e3])

    return eul



def expmap2rotmat(r):

    theta = np.linalg.norm(r)

    r0 = np.divide(r, theta+np.finfo(np.float32).eps)

    r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3,3)

    r0x = r0x-r0x.T

    R = np.eye(3,3)+np.sin(theta)*r0x+(1-np.cos(theta))*(r0x).dot(r0x)

    return R



def unnormalize_data(data, data_mean, data_std, dim_ignore, dim_use, dim_zero):

    t, d = data.shape[0], data_mean.shape[0]   # t = 25, d = 99

    orig_data = np.zeros((t, d), dtype=np.float32)    # [25, 99]

    mask = np.ones((t, d), dtype=np.float32) * 0.7

    dim_use, dim_zero = np.array(dim_use), np.array(dim_zero)

    orig_data[:,dim_use] = data

    mask[:,dim_zero] = 0

    orig_data = np.multiply(orig_data, mask)



    std_mat = np.repeat(data_std.reshape((1,d)), t, axis=0)

    mean_mat = np.repeat(data_mean.reshape((1,d)), t, axis=0)

    orig_data = np.multiply(orig_data, std_mat)+mean_mat

    return orig_data



def normalize_data(data_set, data_mean, data_std, dim_to_use):

    data_out = {}

    for key in data_set.keys():

        data_out[key] = np.divide((data_set[key]-data_mean), data_std)

        data_out[key] = data_out[key][:,dim_to_use]

    return data_out



def normalization_stats(complete_data):

    data_mean = np.mean(complete_data, axis=0)

    data_std = np.std(complete_data, axis=0)

    dimensions_is_zero = []

    dimensions_is_zero.extend(list(np.where(data_std < 1e-4)[0]))

    dimensions_nonzero = []

    dimensions_nonzero.extend(list(np.where(data_std >= 1e-4)[0]))

    data_std[dimensions_is_zero] = 1.0



    dim_to_ignore = [0,  1,  2,  3,  4,  5,  18, 19, 20, 33, 34, 35, 48, 49, 50, 63, 64, 65,

                     66, 67, 68, 69, 70, 71, 72, 73, 74, 87, 88, 89,

                     90, 91, 92, 93, 94, 95, 96, 97, 98]

    dim_to_use = [6,  7,  8,  9,  10, 11, 12, 13, 14,

                  15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,

                  36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51, 52, 53,

                  54, 55, 56, 57, 58, 59, 60 ,61, 62, 75, 76, 77, 78, 79, 80,

                  81, 82, 83, 84, 85, 86]

    return data_mean, data_std, dim_to_ignore, dim_to_use, dimensions_is_zero, dimensions_nonzero



def loss_l1(pred, target, mask=None):

    dist = torch.abs(pred-target).mean(-1).mean(1).mean(0)

    if mask is not None:

        dist = dist * mask

    loss = torch.mean(dist)

    return loss





def adjust_learning_rate(optimizer, epoch, args, freq=300):

    if (epoch-1)%freq == 0:

        lr = args.learning_rate * (0.1**((epoch-1)/freq))

        for param_group in optimizer.param_groups:

            param_group['lr'] = lr



def cal_MAE(targets, outputs, test_sample_num, target_seq_len, data_mean, data_std, dim_ignore, dim_use, dim_zero, dim_nonzero):



    mean_errors = np.zeros((test_sample_num, target_seq_len), dtype=np.float32)

    for i in np.arange(test_sample_num):



        output = outputs[i]  # output: [V, t, d] = [20, 10, 3]

        V, t, d = output.shape

        output = output.permute(1, 0, 2).contiguous().view(t, V * d)

        output_denorm = unnormalize_data(output.cpu().numpy(), data_mean, data_std,

                                         dim_ignore, dim_use, dim_zero)

        t, D = output_denorm.shape

        output_euler = np.zeros((t, D), dtype=np.float32)  # [21, 99]

        for j in np.arange(t):

           for k in np.arange(3, 97, 3):

               output_euler[j, k:k + 3] = rotmat2euler(expmap2rotmat(output_denorm[j, k:k + 3]))



        target = targets[i]

        target = target.permute(1, 0, 2).contiguous().view(t, V * d)

        target_denorm = unnormalize_data(target.cpu().numpy(), data_mean, data_std,

                                         dim_ignore, dim_use, dim_zero)



        target_euler = np.zeros((t, D), dtype=np.float32)

        for j in np.arange(t):

           for k in np.arange(3, 97, 3):

               target_euler[j, k:k + 3] = rotmat2euler(expmap2rotmat(target_denorm[j, k:k + 3]))



        target_denorm[:, 0:6] = 0

        idx_to_use1 = np.where(np.std(target_denorm, 0) > 1e-4)[0]

        idx_to_use2 = dim_nonzero

        idx_to_use = idx_to_use1[np.in1d(idx_to_use1, idx_to_use2)]

        target = target_denorm[:, idx_to_use]
        output = output_denorm[:, idx_to_use]
        t = target.reshape(25 * 48)
        o = output.reshape(25 * 48)

        print(r2_score(t, o), mean_absolute_error(t, o), max_error(t, o))

        # euc_error = np.power(target_denorm[:, idx_to_use] - output_denorm[:, idx_to_use], 2)

        # euc_error = np.sqrt(np.sum(euc_error, 1))



        # mean_errors[i, :euc_error.shape[0]] = euc_error

    #mean_mean_errors = np.mean(np.array(mean_errors), 0)



    return t # mean_mean_errors


def cal_bone(x_joint):
    x_bone_all = []

    neighbor_link_ = [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (1, 9), (5, 9),

                      (9, 10), (10, 11), (11, 12), (10, 13), (13, 14), (14, 15),

                      (15, 16), (10, 17), (17, 18), (18, 19), (19, 20)]

    neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_link_]

    for i, link in enumerate(neighbor_link):
        a = x_joint[:, :, :, link[1]] - x_joint[:, :, :, link[0]]

        x_bone_all.append(a)

    x_bone = torch.stack(x_bone_all, dim=3)

    return x_bone