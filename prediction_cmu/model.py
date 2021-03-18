import torch
import torch.nn as nn
from graph import Graph

class EN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A_j, A_b, A_s, A_t, stride=2, temp_kernel_size=7, residual=True):
        super(EN_Block, self).__init__()
        self.DA_GCO = DA_GCO(in_channels, out_channels, A_j, A_b, A_s, A_t)
        self.TUO = TUO(out_channels, out_channels, kernel_size=temp_kernel_size, stride=stride)

        #  residual module
        if not residual:
            self.residual = lambda fv, fe: (0, 0)
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda fv, fe: (fv, fe)
        else:
            self.residual = TUO(in_channels, out_channels, kernel_size=temp_kernel_size, stride=stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, j_in, b_in):
        j_res, b_res = self.residual(j_in, b_in)
        j_up_in, b_up_in = self.DA_GCO(j_in, b_in)
        j_up_inputs, b_up_inputs = self.TUO(j_up_in, b_up_in)
        j_up_inputs += j_res
        b_up_inputs += b_res
        return j_up_inputs, b_up_inputs

class DA_GCO(nn.Module):
    def __init__(self, in_channels, out_channels, A_j, A_b, A_s, A_t):
        super(DA_GCO, self).__init__()
        self.gcn_joint = GCN(in_channels, A_j)
        self.gcn_bone = GCN(in_channels, A_b)
        self.Update = Update(in_channels, out_channels, A_s, A_t)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, j_inputs, b_inputs):
        d_joint = self.gcn_joint(j_inputs)
        d_bone = self.gcn_bone(b_inputs)
        j_up, b_up = self.Update(d_joint, d_bone)
        return j_up, b_up

class GCN(nn.Module):
    def __init__(self, in_channels, A):
        super().__init__()
        self.A = nn.Parameter(A)
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=in_channels,
                              kernel_size=1,
                              padding=0,
                              stride=1,
                              dilation=1,
                              bias=True)
        self.bn = nn.BatchNorm2d(in_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, i):
        x = self.bn(self.conv(i))
        n, kc, t, v = x.size()
        x = x.reshape(n, -1, v)
        x = torch.einsum('nce,ev->ncv', (x, self.A))
        x.contiguous()
        x = x.view(n, kc, t, v)
        return x

class Update(nn.Module):
    def __init__(self, in_channels, out_channels, A_s, A_t):
        super().__init__()

        self.A_s = nn.Parameter(A_s)
        self.A_t = nn.Parameter(A_t)
        # Updating functions
        self.H_v = nn.Linear(3 * in_channels, out_channels)
        self.H_e = nn.Linear(3 * in_channels, out_channels)
        self.bn_v = nn.BatchNorm2d(out_channels)
        self.bn_e = nn.BatchNorm2d(out_channels)
        bn_init(self.bn_v, 1)
        bn_init(self.bn_e, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fj, fb):
        _, _, _, V_node = fj.shape
        N, C, T, V_edge = fb.shape
        fj = fj.view(N, -1, V_node).contiguous()
        fb = fb.view(N, -1, V_edge).contiguous()

        fj_in_agg = torch.einsum('nce,ev->ncv', fb, self.A_s)
        fj_out_agg = torch.einsum('nce,ev->ncv', fb, self.A_t)
        fjp = torch.stack((fj, fj_in_agg, fj_out_agg), dim=1)
        fjp = fjp.view(N, 3 * C, T, V_node).contiguous().permute(0, 2, 3, 1)
        fjp = self.H_v(fjp).permute(0, 3, 1, 2)
        fjp = self.relu(self.bn_v(fjp))

        fb_in_agg = torch.einsum('nce,ev->ncv', fj, self.A_s.transpose(0, 1))
        fb_out_agg = torch.einsum('nce,ev->ncv', fj, self.A_t.transpose(0, 1))
        fbp = torch.stack((fb, fb_in_agg, fb_out_agg), dim=1)
        fbp = fbp.view(N, 3 * C, T, V_edge).contiguous().permute(0, 2, 3, 1)
        fbp = self.H_v(fbp).permute(0, 3, 1, 2)
        fbp = self.relu(self.bn_v(fbp))
        return fjp, fbp

class TUO(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=2):
        super().__init__()
        # NOTE: assuming that temporal convs are shared between node/edge features
        self.tempconv = TemporalConv(in_channels, out_channels, kernel_size, stride)

    def forward(self, fv, fe):
        return self.tempconv(fv), self.tempconv(fe)

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=2):
        super().__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),   # Conv along the temporal dimension only
            padding=(pad, 0),
            stride=(stride, 1)
        )

        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

def conv_init(conv):
    # nn.init.normal_(0.0, 0.02)
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    # nn.init.normal_(1.0, 0.02)
    nn.init.constant_(bn.bias, 0)

class Model(nn.Module):
    def __init__(self, target_len):
        super(Model, self).__init__()
        self.target_len = target_len
        G = Graph()
        # load adjacent matrix
        #  A_j: joint adjacent matrix
        #  A_b: bone adjacent matrix
        #  A_s: source adjacent matrix
        #  A_t: target adjacent matrix
        self.A_j = torch.from_numpy(G.A_j.astype('float32')).cuda("cuda:0")
        self.A_b = torch.from_numpy(G.A_b.astype('float32')).cuda("cuda:0")
        self.A_s = torch.from_numpy(G.A_s.astype('float32')).cuda("cuda:0")
        self.A_t = torch.from_numpy(G.A_t.astype('float32')).cuda("cuda:0")

        self.EN_Block1 = EN_Block(3, 32, self.A_j, self.A_b, self.A_s, self.A_t, stride=2, residual=False)
        self.EN_Block2 = EN_Block(32, 64, self.A_j, self.A_b, self.A_s, self.A_t, stride=2)
        self.EN_Block3 = EN_Block(64, 128, self.A_j, self.A_b, self.A_s, self.A_t, stride=2)
        self.EN_Block4 = EN_Block(128, 128, self.A_j, self.A_b, self.A_s, self.A_t, stride=2)
        self.EN_Block5 = EN_Block(128, 256, self.A_j, self.A_b, self.A_s, self.A_t, stride=2)
        self.EN_Block6 = EN_Block(256, 256, self.A_j, self.A_b, self.A_s, self.A_t, stride=2)

        self.DA_GCO = DA_GCO(256, 256, self.A_j, self.A_b, self.A_s, self.A_t)

        self.input_r = nn.Linear(3, 256, bias=True)
        self.input_i = nn.Linear(3, 256, bias=True)
        self.input_n = nn.Linear(3, 256, bias=True)

        self.hidden_r = nn.Linear(256, 256, bias=False)
        self.hidden_i = nn.Linear(256, 256, bias=False)
        self.hidden_h = nn.Linear(256, 256, bias=False)

        self.out_fc1 = nn.Linear(256, 256)
        self.out_fc2 = nn.Linear(256, 256)
        self.out_fc3 = nn.Linear(256, 3)

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, j_inputs, b_inputs, last_state, last_sec_state):
        # Encoder
        j_inputs1, b_inputs1 = self.EN_Block1(j_inputs, b_inputs)
        j_inputs2, b_inputs2 = self.EN_Block2(j_inputs1, b_inputs1)
        j_inputs3, b_inputs3 = self.EN_Block3(j_inputs2, b_inputs2)
        j_inputs4, b_inputs4 = self.EN_Block4(j_inputs3, b_inputs3)
        j_inputs5, b_inputs5 = self.EN_Block5(j_inputs4, b_inputs4)
        j_inputs6, b_inputs6 = self.EN_Block6(j_inputs5, b_inputs5)
        j_inputs7, b_inputs7 = self.EN_Block6(j_inputs6, b_inputs6)

        # Decoder
        pred_all = []
        diff_all = []
        last_diff = last_state - last_sec_state
        last_diff = last_diff.permute(0, 2, 1).contiguous()
        last_state = last_state.permute(0, 2, 1).contiguous()
        j_H, b_H = j_inputs7, b_inputs7
        _, _, _, V = j_H.size()
        for step in range(self.target_len):
            if step == 0:
                j_H, b_H = self.DA_GCO(j_H, b_H)
            else:
                last_diff = diff_all[-1]
                last_state = pred_all[-1]
                H = H.permute(0, 2, 1).contiguous()
                H = H.unsqueeze(2)
                j_H = H[:, :, :, 0: V]
                b_H = H[:, :, :, V:]
                j_H, b_H = self.DA_GCO(j_H, b_H)

            j_H = j_H.squeeze(2)
            b_H = b_H.squeeze(2)
            H = torch.cat([j_H, b_H], dim=2)
            H = H.permute(0, 2, 1).contiguous()
            r = torch.sigmoid(self.input_r(last_diff) + self.hidden_r(H))
            z = torch.sigmoid(self.input_i(last_diff) + self.hidden_i(H))
            n = torch.tanh(self.input_n(last_diff) + r * self.hidden_h(H))
            H = (1 - z) * n + z * H

            hidd = self.dropout1(self.leaky_relu(self.out_fc1(H)))
            hidd = self.dropout2(self.leaky_relu(self.out_fc2(hidd)))
            pred = self.out_fc3(hidd)
            pred_ = last_diff + pred
            diff_all.append(pred_)
            pred_state = pred_ + last_state
            pred_all.append(pred_state)

        pred_all = torch.stack(pred_all, dim=1)
        pred_all = pred_all.permute(0, 2, 1, 3).contiguous()
        j_pred_all = pred_all[:, 0: V, :, :]

        return j_pred_all