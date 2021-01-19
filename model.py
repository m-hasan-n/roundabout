from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import outputActivation


def make_mlp_reduced(dim_in, dim_out, batch_norm):
    if batch_norm:
        layers = [nn.Linear(dim_in, dim_out), nn.BatchNorm1d(dim_out), nn.ReLU()]
    else:
        layers = [nn.Linear(dim_in, dim_out), nn.ReLU()]
    return nn.Sequential(*layers)

class roundNet(nn.Module):

    ## Initialization
    def __init__(self, args):
        super(roundNet, self).__init__()

        ## Unpack arguments
        self.args = args

        self.ip_dim = args['ip_dim']
        self.gauss_red = args['Gauss_reduced']

        self.use_intention = args['use_intention']
        self.use_en_ex = args['use_entry_exit_int']

        ## Use gpu flag
        self.use_cuda = args['use_cuda']

        # Batch size
        self.batch_size = args['batch_size']

        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = args['train_flag']

        ## Sizes of network layers
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']

        self.dyn_embedding_size = args['dyn_embedding_size']
        self.input_embedding_size = args['input_embedding_size']

        ## Define network weights
        # Input embedding layer
        self.ip_emb = torch.nn.Linear(self.ip_dim, self.input_embedding_size)

        # Encoder LSTM
        self.enc_lstm = torch.nn.LSTM(self.input_embedding_size, self.encoder_size, 1)

        # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(self.encoder_size, self.dyn_embedding_size)

        #Pooling
        self.bottleneck_dim = args['bottleneck_dim']
        self.batch_norm = args['batch_norm']
        self.soc_embedding_size = self.bottleneck_dim
        self.mlp_pre_dim = 2 * self.encoder_size
        self.rel_pos_embedding = nn.Linear(self.ip_dim, self.encoder_size)
        self.mlp_pre_pool = make_mlp_reduced(self.mlp_pre_dim, self.bottleneck_dim, self.batch_norm)


        # Output layers:
        if self.ip_dim==2:
            op_gauss_dim = 5
        elif self.ip_dim==3:
            if self.gauss_red:
                op_gauss_dim = 7
            else:
                op_gauss_dim=9

        self.op = torch.nn.Linear(self.decoder_size, op_gauss_dim)

        if self.use_intention:
            self.num_lat_classes = args['num_lat_classes']
            self.num_lon_classes = args['num_lon_classes']


            self.op_lat = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lat_classes)
            self.op_lon = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lon_classes)
            self.dec_ip_size = self.soc_embedding_size + self.dyn_embedding_size + self.num_lat_classes + self.num_lon_classes
            if self.use_en_ex:
                self.num_en_ex_classes = args['num_en_ex_classes']
                self.dec_ip_size = self.dec_ip_size + self.num_en_ex_classes
                self.op_en_ex = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_en_ex_classes)
            else:
                self.num_en_ex_classes = 0
        else:
            # Decoder LSTM
            self.dec_ip_size = self.soc_embedding_size + self.dyn_embedding_size

        self.dec_lstm = torch.nn.LSTM(self.dec_ip_size, self.decoder_size)
        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)


    ## Forward Pass
    def forward(self, hist, nbrs, nbr_list_len, lat_enc, lon_enc, en_ex_enc):
        ## Forward pass hist:
        _, (hist_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))
        hist_enc = self.leaky_relu(self.dyn_emb(hist_enc.view(hist_enc.shape[1], hist_enc.shape[2])))

        ## Forward pass nbrs
        _, (nbrs_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])

        ## Pooling the Nbrs hist
        soc_enc = torch.zeros(nbr_list_len.shape[0], self.bottleneck_dim).float()

        if self.use_cuda:
            soc_enc = soc_enc.cuda()
        cntr = 0
        for ind in range(nbr_list_len.shape[0]):
            no_nbrs = int(nbr_list_len[ind].item())
            if no_nbrs > 0:
                curr_nbr_pos = nbrs[:, cntr:cntr + no_nbrs, :]
                curr_nbr_enc = nbrs_enc[cntr:cntr + no_nbrs, :]
                cntr += no_nbrs

                end_nbr_pose = curr_nbr_pos[-1]
                rel_pos_embedding = self.rel_pos_embedding(end_nbr_pose)
                mlp_h_input = torch.cat([rel_pos_embedding, curr_nbr_enc], dim=1)

                # if only 1 neighbor, BatchNormalization will not work
                # So calling model.eval() before feeding the data will change
                # the behavior of the BatchNorm layer to use the running estimates
                # instead of calculating them
                if mlp_h_input.shape[0] == 1 & self.batch_norm:
                    self.mlp_pre_pool.eval()

                curr_pool_h = self.mlp_pre_pool(mlp_h_input)

                curr_pool_h = curr_pool_h.max(0)[0]
                soc_enc[ind] = curr_pool_h

        ## Concatenate encodings:
        enc = torch.cat((soc_enc, hist_enc), 1)

        if self.use_intention:
            ## Maneuver recognition:
            lat_pred = self.softmax(self.op_lat(enc))
            lon_pred = self.softmax(self.op_lon(enc))
            if self.use_en_ex:
                en_ex_pred = self.softmax(self.op_en_ex(enc))

            if self.train_flag:
                if self.use_en_ex:
                    ## Concatenate maneuver encoding of the true maneuver
                    enc = torch.cat((enc, lat_enc, lon_enc, en_ex_enc), 1)
                    fut_pred = self.decode(enc)
                    return fut_pred, lat_pred, lon_pred, en_ex_pred
                else:
                    enc = torch.cat((enc, lat_enc, lon_enc), 1)
                    fut_pred = self.decode(enc)
                    return fut_pred, lat_pred, lon_pred

            else:
                fut_pred = []
                ## Predict trajectory distributions for each maneuver class
                for k in range(self.num_lon_classes):
                    for l in range(self.num_lat_classes):
                        # for m in range(self.num_en_ex_classes):
                            lat_enc_tmp = torch.zeros_like(lat_enc)
                            lon_enc_tmp = torch.zeros_like(lon_enc)
                            lat_enc_tmp[:, l] = 1
                            lon_enc_tmp[:, k] = 1

                            if self.use_en_ex:
                                en_ex_enc_tmp = torch.zeros_like(en_ex_enc)
                                en_ex_enc_tmp[:, m] =1
                                enc_tmp = torch.cat((enc, lat_enc_tmp, lon_enc_tmp, en_ex_enc_tmp), 1)
                                fut_pred.append(self.decode(enc_tmp))

                            else:
                                enc_tmp = torch.cat((enc, lat_enc_tmp, lon_enc_tmp),1)
                                fut_pred.append(self.decode(enc_tmp))
                if self.use_en_ex:
                    return fut_pred, lat_pred, lon_pred, en_ex_pred
                else:
                    return fut_pred, lat_pred, lon_pred
        else:
            fut_pred = self.decode(enc)
            return fut_pred


    def decode(self, enc):
        enc = enc.repeat(self.out_length, 1, 1)
        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred)
        return fut_pred



