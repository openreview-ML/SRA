import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from torchvision import models
import scipy.sparse as sp

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes / stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=32, nc=3):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(1000, 512)
        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

        self.dropout = nn.Dropout(p=0.2)
    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=7)
        x = self.layer4(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer1(x)
        x = F.interpolate(x, size=(112, 112), mode='bilinear')
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), 3, 224, 224)
        return x
class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator_conv(nn.Module):
    """Fully convolutional Generator network if latent are cubic."""
    def __init__(self, nc=3, conv_dim=64, repeat_num=2):
        super(Generator_conv, self).__init__()
        '''
        encoder
        '''
        self.start_layers = []
        # self.start_layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        self.start_layers.append(nn.Conv2d(nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        self.start_layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        self.start_layers.append(nn.ReLU(inplace=True))
        self.start_part = nn.Sequential(*self.start_layers)

        # Down-sampling layers.
        self.down_layers = []
        curr_dim = conv_dim
        for i in range(2):
            self.down_layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            self.down_layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            self.down_layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2
        self.down_part = nn.Sequential(*self.down_layers)
        self.eli_pose_part = nn.Sequential(*self.start_layers, *self.down_layers)

        # Bottleneck layers.
        self.bottle_encoder_layers = []
        for i in range(repeat_num):
            self.bottle_encoder_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.bottlen_encoder_part = nn.Sequential(*self.bottle_encoder_layers)

        self.encoder = nn.Sequential(*self.start_layers, *self.down_layers, *self.bottle_encoder_layers)
        '''
        decoder
        '''
        self.bottle_decoder_layers = []
        for i in range(repeat_num):
            self.bottle_decoder_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.bottlen_decoder_part = nn.Sequential(*self.bottle_decoder_layers)

        # Up-sampling layers.
        self.up_layers = []
        for i in range(2):
            if i ==0:
                self.up_layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            else:
                self.up_layers.append(
                    nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))

            self.up_layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            self.up_layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.up_layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        self.up_layers.append(nn.Tanh())
        self.up_part = nn.Sequential(*self.up_layers)

        self.decoder = nn.Sequential(*self.bottle_decoder_layers, *self.up_layers)

        self.main = nn.Sequential(*self.start_layers, *self.down_layers, *self.bottle_encoder_layers, *self.bottle_decoder_layers, *self.up_layers)

    def forward(self, x):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or
        x1 = self.encoder(x)
        x2 = self.decoder(x1)

        return x2, x1



    def forward_origin(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)
class Generator_fc(nn.Module):
    """Generator network, with fully connected layers to get latent Z"""
    def __init__(self, nc=3, conv_dim=64, repeat_num=2, z_dim=500, graph_init=None,gr_emb = None):
        self.z_dim = z_dim
        super(Generator_fc, self).__init__()
        '''
        encoder
        '''
        self.encoder = models.resnet18(pretrained=True)
        path = graph_init
        graph = torch.load(path)
        self.embeddings = graph['embeddings'].cuda()
        adj = graph['adj']
        self.dropout = nn.Dropout(p=0.2)
        self.fc_encoder = nn.Sequential(
            nn.Linear(1000,self.z_dim),
            nn.ReLU(True)
        )
        '''
        decoder
        '''
        self.fc_decoder = nn.Sequential(
            nn.Linear(self.z_dim,1000),
            nn.ReLU(True)
        )

        hidden_layers = gr_emb
        self.gcn = GCN(adj, self.embeddings.shape[1],self.z_dim, hidden_layers)


        # self.fc_sem_encoder = nn.Sequential(
        #     nn.Linear(600, 1024),
        #     nn.ReLU(True),
        #     nn.Linear(1024, self.z_dim)
        # )
        #
        # '''
        # decoder
        # '''
        # self.fc_sem_decoder = nn.Sequential(
        #     nn.Linear(self.z_dim, 1024),
        #     nn.ReLU(True),
        #     nn.Linear(1024, 600)
        # )



        # Decoder Bottleneck layers.
        self.decoder = ResNet18Dec()
        #self.main = nn.Sequential(*self.start_layers, *self.encoder, *self.decoder,*self.decoder)

        # # fc layers
        # self.fc_encoder = nn.Sequential(
        #     nn.Linear(256 * 14 * 14, 4096),
        #     nn.ReLU(True),
        #     nn.Linear(4096, self.z_dim)
        # )
        # '''
        # decoder
        # '''
        # self.fc_decoder = nn.Sequential(
        #     nn.Linear(self.z_dim, 4096),
        #     nn.ReLU(True),
        #     nn.Linear(4096, 256 * 14 * 14)
        # )
        #

        # #
        # # self.fc_sem_encoder = nn.Sequential(
        # #     nn.Linear(600, 1024),
        # #     nn.ReLU(True),
        # #     nn.Linear(1024, self.z_dim)
        # # )
        # #
        # # '''
        # # decoder
        # # '''
        # # self.fc_sem_decoder = nn.Sequential(
        # #     nn.Linear(self.z_dim, 1024),
        # #     nn.ReLU(True),
        # #     nn.Linear(1024, 600)
        # # )
        #
        #
        #
        # # Decoder Bottleneck layers.
        # self.bottle_decoder_layers = []
        # for i in range(repeat_num):
        #     self.bottle_decoder_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        # self.bottlen_decoder_part = nn.Sequential(*self.bottle_decoder_layers)
        #
        # # Up-sampling layers.
        # self.up_layers = []
        # for i in range(4):
        #     if i <= 1:
        #         self.up_layers.append(nn.ConvTranspose2d(curr_dim, curr_dim, kernel_size=4, stride=2, padding=1, bias=False))
        #         self.up_layers.append(nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True))
        #         self.up_layers.append(nn.ReLU(inplace=True))
        #     else:
        #         self.up_layers.append(
        #             nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
        #         self.up_layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
        #         self.up_layers.append(nn.ReLU(inplace=True))
        #         curr_dim = curr_dim // 2
        #
        #
        # self.up_layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        # self.up_layers.append(nn.Tanh())
        # self.up_part = nn.Sequential(*self.up_layers)
        #
        # self.decoder = nn.Sequential(*self.bottle_decoder_layers, *self.up_layers)

    def forward(self, x):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or
        x1 = self.encoder(x)
        x1 = x1.view(x.shape[0], -1)
        z1= self.fc_encoder(x1)
        z = self.dropout(z1)
        x2 = self.fc_decoder(z)
        x2 = x2.view(x.shape[0], 1000)
        x3 = self.decoder(x2)

        return x3, z

    def forward_origin(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)

class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, relu=True):
        super().__init__()

        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None

        self.layer = nn.Linear(in_channels, out_channels)

        if relu:
            # self.relu = nn.LeakyReLU(negative_slope=0.2)
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, inputs, adj):
        if self.dropout is not None:
            inputs = self.dropout(inputs)

        outputs = torch.mm(adj, torch.mm(inputs, self.layer.weight.T)) + self.layer.bias

        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs


class GCN(nn.Module):

    def __init__(self, adj, in_channels, out_channels, hidden_layers):
        super().__init__()

        adj = normt_spm(adj, method='in')
        adj = spm_to_tensor(adj)
        self.adj = adj.cuda()

        self.train_adj = self.adj

        hl = hidden_layers.split(',')
        if hl[-1] == 'd':
            dropout_last = True
            hl = hl[:-1]
        else:
            dropout_last = False

        i = 0
        layers = []
        last_c = in_channels
        for c in hl:
            if c[0] == 'd':
                dropout = True
                c = c[1:]
            else:
                dropout = False
            c = int(c)

            i += 1
            conv = GraphConv(last_c, c, dropout=dropout)
            self.add_module('conv{}'.format(i), conv)
            layers.append(conv)

            last_c = c

        conv = GraphConv(last_c, out_channels, relu=False, dropout=dropout_last)
        self.add_module('conv-last', conv)
        layers.append(conv)

        self.layers = layers

    def forward(self, x):
        if self.training:
            for conv in self.layers:
                x = conv(x, self.train_adj)
        else:
            for conv in self.layers:
                x = conv(x, self.adj)
        return F.normalize(x)

def normt_spm(mx, method='in'):
    if method == 'in':
        mx = mx.transpose()
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    if method == 'sym':
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = mx.dot(r_mat_inv).transpose().dot(r_mat_inv)
        return mx

def spm_to_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack(
            (sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
class Generator_fc_dsprites(nn.Module):
    """Generator network, with fully connected layers"""
    def __init__(self, nc=3, conv_dim=64, repeat_num=2, z_dim=500):
        self.z_dim = z_dim
        super(Generator_fc_dsprites, self).__init__()
        '''
        encoder
        '''
        self.start_layers = []
        # self.start_layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        self.start_layers.append(nn.Conv2d(nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        self.start_layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        self.start_layers.append(nn.ReLU(inplace=True))
        self.start_part = nn.Sequential(*self.start_layers)

        # Down-sampling layers.
        self.down_layers = []
        curr_dim = conv_dim
        for i in range(4):
            if i <= 1:
                self.down_layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
                self.down_layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
                self.down_layers.append(nn.ReLU(inplace=True))
                curr_dim = curr_dim * 2
            else:
                self.down_layers.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=4, stride=2, padding=1, bias=False))
                self.down_layers.append(nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True))
                self.down_layers.append(nn.ReLU(inplace=True))

        self.down_part = nn.Sequential(*self.down_layers)
        self.eli_pose_part = nn.Sequential(*self.start_layers, *self.down_layers)

        # Bottleneck layers.
        self.bottle_encoder_layers = []
        for i in range(repeat_num):
            self.bottle_encoder_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.bottlen_encoder_part = nn.Sequential(*self.bottle_encoder_layers)

        self.encoder = nn.Sequential(*self.start_layers, *self.down_layers, *self.bottle_encoder_layers)
        # fc layers
        self.fc_encoder = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(True),
            nn.Linear(512, self.z_dim)
        )
        '''
        decoder
        '''
        self.fc_decoder = nn.Sequential(
            nn.Linear(self.z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 256 * 4 * 4)
        )
        self.bottle_decoder_layers = []
        for i in range(repeat_num):
            self.bottle_decoder_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.bottlen_decoder_part = nn.Sequential(*self.bottle_decoder_layers)

        # Up-sampling layers.
        self.up_layers = []
        for i in range(4):
            if i <= 1:
                self.up_layers.append(nn.ConvTranspose2d(curr_dim, curr_dim, kernel_size=4, stride=2, padding=1, bias=False))
                self.up_layers.append(nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True))
                self.up_layers.append(nn.ReLU(inplace=True))
            else:
                self.up_layers.append(
                    nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
                self.up_layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
                self.up_layers.append(nn.ReLU(inplace=True))
                curr_dim = curr_dim // 2


        self.up_layers.append(nn.Conv2d(curr_dim, nc, kernel_size=7, stride=1, padding=3, bias=False))
        # self.up_layers.append(nn.Tanh())
        self.up_part = nn.Sequential(*self.up_layers)

        self.decoder = nn.Sequential(*self.bottle_decoder_layers, *self.up_layers)

        self.main = nn.Sequential(*self.start_layers, *self.down_layers, *self.bottle_encoder_layers, *self.bottle_decoder_layers, *self.up_layers)

    def forward(self, x):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or
        x1 = self.encoder(x)
        x1 = x1.view(x.shape[0], -1)
        z= self.fc_encoder(x1)
        x2 = self.fc_decoder(z)
        x2 = x2.view(x.shape[0], 256, 4, 4)
        x3 = self.decoder(x2)

        return x3, z



    def forward_origin(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)

class Discriminator_multi(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator_multi, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        return out_src
class Discriminator_multi_origin(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator_multi, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        return out_src

class Discriminator(nn.Module):
        """Discriminator network with PatchGAN."""

        def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
            super(Discriminator, self).__init__()
            layers = []
            layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))

            curr_dim = conv_dim
            for i in range(1, repeat_num):
                layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
                layers.append(nn.LeakyReLU(0.01))
                curr_dim = curr_dim * 2

            kernel_size = int(image_size / np.power(2, repeat_num))
            self.main = nn.Sequential(*layers)
            self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

        def forward(self, x):
            h = self.main(x)
            out_src = self.conv1(h)
            return out_src
            # out_cls = self.conv2(h)
            # return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class Discriminator_pose(nn.Module):
    """For pose info elimination, Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6, imput_dim=256):
        super(Discriminator_pose, self).__init__()
        layers = []
        conv_dim = conv_dim*2*2
        #　eliminate the last 6 dim to store the pose information
        # layers.append(nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Conv2d(imput_dim, conv_dim*2, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim*2
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        # kernel_size = 2
        self.main = nn.Sequential(*layers)
        # self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        # out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_cls.view(out_cls.size(0), out_cls.size(1))
class Discriminator_pose_softmax(nn.Module):
    """For pose info elimination, Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator_pose, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2


        layers.append(nn.Linear(np.square(image_size//(2 * (repeat_num + 1))) * curr_dim, 120))
        layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.Linear(120, c_dim))
        layers.append(nn.LeakyReLU(0.01))
        self.main = nn.Sequential(*layers)


    def forward(self, x):
        h = self.main(x)
        # out_src = self.conv1(h)
        # out_cls = self.conv2(h)
        # return out_cls.view(out_cls.size(0), out_cls.size(1))
        return F.softmax(h)
