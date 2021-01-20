from __future__ import print_function
import torch
from model import roundNet
from utils import roundDataset, maskedNLL, maskedMSETest, maskedNLLTest, saveResultFiles, maskedMSETest_XY
from torch.utils.data import DataLoader
import time
import numpy as np


# REPRODUCIBILITY
# torch.manual_seed(0)
# np.random.seed(0)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# # torch.set_deterministic(True)
# #for lstm
# CUDA_LAUNCH_BLOCKING=1


## Network Arguments
args = {}
args['use_cuda'] = True
args['ip_dim'] = 3
args['Gauss_reduced'] = True
args['encoder_size'] = 32
args['decoder_size'] = 64
args['in_length'] = 13
args['out_length'] = 25
args['dyn_embedding_size'] = 16
args['input_embedding_size'] = 16
args['train_flag'] = False
args['batch_size'] = 128
args['bottleneck_dim'] = 64
args['batch_norm'] = True

args['use_intention'] = True
if args['use_intention']:
    args['num_lat_classes'] = 8
    args['num_lon_classes'] = 3

args['use_entry_exit_int'] = False
if args['use_entry_exit_int']:
    args['num_en_ex_classes'] = 2

# Initialize network
net = roundNet(args)

# load the trained model
# net_fname = 'trained_models/round_baseline.tar'
net_fname = 'trained_models/round_3D_Intention_Anchors_ref.tar'

if (args['use_cuda']):
    net.load_state_dict(torch.load(net_fname), strict=False)
else:
    net.load_state_dict(torch.load(net_fname , map_location= lambda storage, loc: storage), strict=False)

## Initialize data loaders
tsSet = roundDataset('data/TestSet.mat')
tsDataloader = DataLoader(tsSet,batch_size=128,shuffle=True,num_workers=8,collate_fn=tsSet.collate_fn)

lossVals = torch.zeros(args['out_length'])
counts = torch.zeros(args['out_length'])

if args['use_cuda']:
    net = net.cuda()
    lossVals = lossVals.cuda()
    counts = counts.cuda()

for i, data in enumerate(tsDataloader):
    hist, nbrs, nbr_list_len, fut, lat_enc, lon_enc, op_mask, ds_ids, vehicle_ids, frame_ids, goal_enc, en_ex_enc  = data

    if args['use_cuda']:
        hist = hist.cuda()
        nbrs = nbrs.cuda()
        nbr_list_len = nbr_list_len.cuda()
        fut = fut.cuda()
        op_mask = op_mask.cuda()
        ds_ids = ds_ids.cuda()
        vehicle_ids = vehicle_ids.cuda()
        frame_ids = frame_ids.cuda()
        lat_enc = lat_enc.cuda()
        lon_enc = lon_enc.cuda()
        goal_enc = goal_enc.cuda()
        en_ex_enc = en_ex_enc.cuda()

    # Forward pass
    if args['use_intention']:
        if args['use_entry_exit_int']:
            fut_pred, lat_pred, lon_pred, en_ex_pred = net(hist, nbrs, nbr_list_len, lat_enc, lon_enc, en_ex_enc)
        else:
            fut_pred, lat_pred, lon_pred = net(hist, nbrs, nbr_list_len, lat_enc, lon_enc, en_ex_enc)

        fut_pred_max = torch.zeros_like(fut_pred[0])
        for k in range(lat_pred.shape[0]):
            lat_man = torch.argmax(lat_pred[k, :]).detach()
            lon_man = torch.argmax(lon_pred[k, :]).detach()

            if args['use_entry_exit_int']:
                en_ex = torch.argmax(en_ex_pred[k, :]).detach()
                indx = lon_man * args['num_lat_classes'] * args['num_en_ex_classes'] + lat_man * args['num_en_ex_classes'] + en_ex
            else:
                indx = lon_man * args['num_lat_classes'] + lat_man

            fut_pred_max[:, k, :] = fut_pred[indx][:, k, :]

        l, c = maskedMSETest(fut_pred_max, fut, op_mask)
    else:
        fut_pred = net(hist, nbrs, nbr_list_len)
        l, c = maskedMSETest(fut_pred, fut, op_mask)

    lossVals += l.detach()
    counts += c.detach()


print(torch.pow(lossVals / counts, 0.5))  # Calculate RMSE
loss_total = torch.pow(lossVals / counts, 0.5)
fname = 'outfiles/rmse_from_code_' + str(args['ip_dim']) +'D_intention_anchors_ref.csv'
rmse_file = open(fname, 'w')
np.savetxt(rmse_file, loss_total.cpu())


