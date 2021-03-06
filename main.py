import numpy as np
from utils_1 import *
import torch
import torch.nn as nn
from model_1 import Model
import torch.optim as optim
import os
from Early_Stop import EarlyStopping



if __name__ == '__main__':

    torch.backends.cudnn.enabled = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dev = "cuda:0"
    data_dir = 'data/h3.6m/dataset'
    subtrain = [1, 6, 7, 8, 9, 11]  # human motion sequences for training
    subtest = [5]  # human motion sequences for test
    actions = ["walking", "eating", "smoking", "discussion", "directions",
               "greeting", "phoning", "posing", "purchases", "sitting", "sittingdown",
               "takingphoto", "waiting", "walkingdog", "walkingtogether"]

    # download and normalize all human motion sequences for training and test
    train_dict, complete_train = load_data(data_dir, subtrain, actions)
    data_mean, data_std, dim_ignore, dim_use, dim_zero, dim_nonzero = normalization_stats(complete_train)
    normed_train_dict = normalize_data(train_dict, data_mean, data_std, dim_use)

    test_dict, complete_test = load_data(data_dir, subtest, actions)
    normed_test_dict = normalize_data(test_dict, data_mean, data_std, dim_use)

    epoch = 10000
    batch_size = 32
    source_seq_len = 50   # length of observation sequence
    target_seq_len = 25   # length of human motions to be predicted
    test_sample_num = 8   # number of test samples for each action
    learning_rate = 0.0001
    step = np.arange(1000, epoch, 1000)    # steps for adjusting learning rate

    # load model
    Model = Model(target_seq_len)
    Model = Model.cuda()
    Model.apply(weights_init)
    # Model_q = nn.DataParallel(Model_q, device_ids=[0, 1])

    # set optimizer
    optimizer = optim.Adam(params=Model.parameters(), lr=learning_rate, weight_decay=0.0001)

    early_stopping = EarlyStopping(patience=25, verbose=True)

    train_loss = []
    test_loss = []
    # training
    # Model.train()
    # for n in range(epoch):
    #     print('---------', n, '----------')
    #     # adjust learning rate
    #
    #     lr = learning_rate * 0.98**np.sum(n >= step)
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    #
    #     #  generate a batch of training samples
    #     j_states, j_last_state, targets = train_sample(normed_train_dict,
    #                                                 batch_size,
    #                                                 source_seq_len,
    #                                                 target_seq_len,
    #                                                 len(dim_use))
    #
    #     j_states = torch.Tensor(j_states).float().to(dev)
    #     j_last_state = torch.Tensor(j_last_state).float().to(dev)
    #     targets = torch.Tensor(targets).float().to(dev)
    #
    #     N, T, D = targets.size()
    #     targets = targets.contiguous().view(N, T, -1, 3).permute(0, 2, 1, 3)
    #
    #     N, T, D = j_states.size()
    #     j_states = j_states.contiguous().view(N, T, -1, 3)
    #     j_states = j_states.permute(0, 3, 1, 2).contiguous()
    #     b_states = cal_bone(j_states)
    #
    #     j_last_state = j_last_state.contiguous().view(N, 1, -1, 3).permute(0, 3, 1, 2)
    #     b_last_state = cal_bone(j_last_state)
    #     last_state = torch.cat([j_last_state, b_last_state], dim=3).squeeze(2)
    #
    #     outputs = Model(j_states, b_states, last_state)
    #
    #     loss = loss_l1(outputs, targets)
    #     optimizer.zero_grad()
    #     loss.backward(retain_graph=True)
    #     nn.utils.clip_grad_norm_(Model.parameters(), 0.5)
    #     optimizer.step()
    #     print(loss.data.item())
    #
    #     if n % 500 == 0:
    #         train_loss.append(loss.data.item())
    #
    #         # test
    #         all_errors = []
    #         for action_num, action in enumerate(actions):
    #             j_states, j_last_state, targets = srnn_sample(normed_test_dict,
    #                                                                 action,
    #                                                                 source_seq_len,
    #                                                                 target_seq_len,
    #                                                                 len(dim_use))
    #
    #             j_states = torch.Tensor(j_states).float().to(dev)
    #             j_last_state = torch.Tensor(j_last_state).float().to(dev)
    #             targets = torch.Tensor(targets).float().to(dev)
    #
    #             N, T, D = targets.size()
    #             targets = targets.contiguous().view(N, T, -1, 3).permute(0, 2, 1, 3)
    #
    #             N, T, D = j_states.size()
    #             j_states = j_states.contiguous().view(N, T, -1, 3)
    #             j_states = j_states.permute(0, 3, 1, 2).contiguous()
    #             b_states = cal_bone(j_states)
    #
    #             j_last_state = j_last_state.contiguous().view(N, 1, -1, 3).permute(0, 3, 1, 2)
    #             b_last_state = cal_bone(j_last_state)
    #             last_state = torch.cat([j_last_state, b_last_state], dim=3).squeeze(2)
    #
    #             with torch.no_grad():
    #                 outputs = Model(j_states, b_states, last_state)
    #
    #             mean_mean_errors = cal_MAE(targets, outputs, test_sample_num, target_seq_len, data_mean,
    #                                        data_std, dim_ignore, dim_use, dim_zero, dim_nonzero)
    #
    #             print(action)
    #             print(mean_mean_errors)
    #             all_errors.append(mean_mean_errors)
    #             print('---------------------------------------')
    #
    #         # early stop
    #         a = np.array(all_errors)
    #         val_loss = np.sum(a, 0)
    #         early_stopping(val_loss, Model)
    #         if early_stopping.early_stop:
    #            print("Early stopping")
    #            break

    # test
    Model.load_state_dict(torch.load('checkpoint_1.pt'))
    for n in range(epoch):
        print('---------', n, '----------')
        # adjust learning rate
        for action_num, action in enumerate(actions):
            j_states, j_last_state, targets = srnn_sample(normed_test_dict,
                                                          action,
                                                          source_seq_len,
                                                          target_seq_len,
                                                          len(dim_use))

            j_states = torch.Tensor(j_states).float().to(dev)
            j_last_state = torch.Tensor(j_last_state).float().to(dev)
            targets = torch.Tensor(targets).float().to(dev)

            N, T, D = targets.size()
            targets = targets.contiguous().view(N, T, -1, 3).permute(0, 2, 1, 3)

            N, T, D = j_states.size()
            j_states = j_states.contiguous().view(N, T, -1, 3)
            j_states = j_states.permute(0, 3, 1, 2).contiguous()
            b_states = cal_bone(j_states)

            j_last_state = j_last_state.contiguous().view(N, 1, -1, 3).permute(0, 3, 1, 2)
            b_last_state = cal_bone(j_last_state)
            last_state = torch.cat([j_last_state, b_last_state], dim=3).squeeze(2)

            with torch.no_grad():
                outputs = Model(j_states, b_states, last_state)
                # print(outputs)

            mean_mean_errors = cal_MAE(targets, outputs, test_sample_num, target_seq_len, data_mean,
                                       data_std, dim_ignore, dim_use, dim_zero, dim_nonzero)

            # print(action)
            # print(mean_mean_errors[1], mean_mean_errors[3], mean_mean_errors[7], mean_mean_errors[9], mean_mean_errors[-1])
            # print('---------------------------------------')
