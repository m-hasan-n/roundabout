from __future__ import print_function
import torch
from model_anchor import roundNet
from utils_anchor import roundDataset, maskedNLL, maskedMSETest, maskedNLLTest, saveResultFiles, maskedMSETest_XY, anchor_inverse, multi_pred
from torch.utils.data import DataLoader
import time
import numpy as np
import scipy.io as scp

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
args['d_s'] = 4

args['num_lat_classes'] = 8
args['num_lon_classes'] = 3

args['anchor_int'] = False


# Initialize network
net = roundNet(args)

# load the trained model
net_fname = 'trained_models/round_3D_Intention_timeChange_latlong_anchor.tar'
if (args['use_cuda']):
    net.load_state_dict(torch.load(net_fname), strict=False)
else:
    net.load_state_dict(torch.load(net_fname , map_location= lambda storage, loc: storage), strict=False)

## Initialize data loaders
tsSet = roundDataset('data/TestSet.mat')
tsDataloader = DataLoader(tsSet,batch_size=128,shuffle=True,num_workers=8,collate_fn=tsSet.collate_fn)
anchor_traj = scp.loadmat('data/TrainSet.mat')['anchor_traj_mean']

lossVals = torch.zeros(args['out_length'])
counts = torch.zeros(args['out_length'])
lossVals2 = torch.zeros(args['out_length'])
counts2 = torch.zeros(args['out_length'])

if args['use_cuda']:
    net = net.cuda()
    lossVals = lossVals.cuda()
    counts = counts.cuda()
    lossVals2 = lossVals2.cuda()
    counts2 = counts2.cuda()



for i, data in enumerate(tsDataloader):

    hist, nbrs, nbr_list_len, fut, lat_enc, lon_enc, op_mask, ds_ids, vehicle_ids, frame_ids, fut_anchred = data

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
        fut_anchred = fut_anchred.cuda()
        lon_enc = lon_enc.cuda()

    fut_pred, lat_pred, lon_pred = net(hist, nbrs, nbr_list_len, lat_enc, lon_enc)

    fut_pred_max = torch.zeros_like(fut_pred[0])
    fut_pred_wt = torch.zeros_like(fut_pred[0])
    for k in range(lat_pred.shape[0]):
        lat_man = torch.argmax(lat_pred[k, :]).detach()
        lon_man = torch.argmax(lon_pred[k, :]).detach()
        indx = lon_man * args['num_lat_classes'] + lat_man
        # indx = lat_man
        fut_pred_max[:, k,:] = fut_pred[indx][:, k, :]
        fut_pred_wt[:, k,:] = multi_pred(lat_pred[k, :], lon_pred[k, :], fut_pred, k, anchor_traj, args['d_s'])



    fut_pred_max = anchor_inverse(fut_pred_max, lat_pred, lon_pred, anchor_traj, args['d_s'], multi=False)
    # fut_pred_wt = anchor_inverse(fut_pred_wt, lat_pred, lon_pred, anchor_traj, args['d_s'], multi=False)

    l, c = maskedMSETest(fut_pred_max, fut, op_mask)
    l2, c2 = maskedMSETest(fut_pred_wt, fut, op_mask)

    lossVals += l.detach()
    counts += c.detach()

    lossVals2 += l2.detach()
    counts2 += c2.detach()

print('regural loss')
print(torch.pow(lossVals / counts, 0.5))  # Calculate RMSE

print('weighted loss')
print(torch.pow(lossVals2 / counts2, 0.5))  # Calculate RMSE


loss_total = torch.pow(lossVals / counts, 0.5)
fname = 'outfiles/rmse_from_code_' + str(args['ip_dim']) +'D_intention_4s_latlon_anchor.csv'
rmse_file = open(fname, 'w')
np.savetxt(rmse_file, loss_total.cpu())




