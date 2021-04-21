from __future__ import print_function
import torch
from model import roundNet
from utils import roundDataset, maskedMSETest, anchor_inverse, multi_pred, horiz_eval
from torch.utils.data import DataLoader
import numpy as np
import scipy.io as scp

## Network Arguments
from model_args import args
args['train_flag'] = False

# Initialize network
net = roundNet(args)

# load the trained model
model_basename = 'round_' + str(args['ip_dim']) + 'D_'
if args['use_intention']:
    model_basename += 'Intention_'
    if args['use_anchors']:
        model_basename += 'Anchors'
net_fname = 'trained_models/' + model_basename + '.tar'

if (args['use_cuda']):
    net.load_state_dict(torch.load(net_fname), strict=False)
else:
    net.load_state_dict(torch.load(net_fname , map_location= lambda storage, loc: storage), strict=False)

## Initialize data loaders
tsSet = roundDataset('data/TestSet.mat')
tsDataloader = DataLoader(tsSet,batch_size=128,shuffle=True,num_workers=8,collate_fn=tsSet.collate_fn)
anchor_traj = scp.loadmat('data/TrainSet.mat')['anchor_traj_mean']

## Initialize loss variables
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
    hist, nbrs, nbr_list_len, fut, lat_enc, lon_enc, op_mask, \
    ds_ids, vehicle_ids, frame_ids, fut_anchred  = data

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
        fut_anchred = fut_anchred.cuda()

    # Forward pass
    if args['use_intention']:

        fut_pred, lat_pred, lon_pred = net(hist, nbrs, nbr_list_len, lat_enc, lon_enc)

        # maximum a posteriori probability (MAP) and 'Weighted' estimates
        fut_pred_max = torch.zeros_like(fut_pred[0])
        fut_pred_wt = torch.zeros_like(fut_pred[0])

        for k in range(lat_pred.shape[0]):
            lat_man = torch.argmax(lat_pred[k, :]).detach()
            lon_man = torch.argmax(lon_pred[k, :]).detach()
            indx = lon_man * args['num_lat_classes'] + lat_man

            fut_pred_max[:, k, :] = fut_pred[indx][:, k, :]
            if args['use_anchors']:
                fut_pred_wt[:, k, :] = multi_pred(lat_pred[k, :], lon_pred[k, :], fut_pred, k, anchor_traj, args['d_s'])

        if args['use_anchors']:
            fut_pred_max = anchor_inverse(fut_pred_max, lat_pred, lon_pred, anchor_traj, args['d_s'], multi=False)
            l2, c2 = maskedMSETest(fut_pred_wt, fut, op_mask)
            lossVals2 += l2.detach()
            counts2 += c2.detach()

        l, c = maskedMSETest(fut_pred_max, fut, op_mask)
    else:
        fut_pred = net(hist, nbrs, nbr_list_len)
        l, c = maskedMSETest(fut_pred, fut, op_mask)

    lossVals += l.detach()
    counts += c.detach()

# Calculate RMSE Loss evaluated on a 4s horizon
pred_horiz = 4
print('MAP RMSE evaluated on a 4s horizon')
MAP_rmse = torch.pow(lossVals / counts, 0.5)
MAP_horiz = horiz_eval(MAP_rmse, pred_horiz)
print(MAP_horiz)

print('Weighted RMSE evaluated on a 4s horizon')
Weighted_rmse = torch.pow(lossVals2 / counts2, 0.5)
Weighted_horiz = horiz_eval(Weighted_rmse, pred_horiz)
print(Weighted_horiz)

#Saving to evaluation files
fname = 'eval_res/' + model_basename + '_MAP.csv'
rmse_file = open(fname, 'ab')
np.savetxt(rmse_file, MAP_horiz.cpu())
rmse_file.close()

fname = 'eval_res/' + model_basename + '_WT.csv'
rmse_file = open(fname, 'ab')
np.savetxt(rmse_file, Weighted_horiz.cpu())
rmse_file.close()