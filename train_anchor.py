from __future__ import print_function
import torch
from model_anchor import roundNet
from utils_anchor import roundDataset, maskedNLL,maskedMSE,maskedNLLTest, maskedNLLTest_LatInt, anchor_inverse
from torch.utils.data import DataLoader
import time
import math
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
args['train_flag'] = True
args['batch_size'] = 128
args['bottleneck_dim'] = 64
args['batch_norm'] = True
args['num_lat_classes'] = 8
args['d_s'] =4

# Initialize network
net = roundNet(args)
if args['use_cuda']:
    net = net.cuda()

## Initialize optimizer
pretrainEpochs = 5
trainEpochs = 3
optimizer = torch.optim.Adam(net.parameters())
batch_size = args['batch_size']
crossEnt = torch.nn.BCELoss()


anchor_traj = scp.loadmat('data/TrainSet.mat')['anchor_traj_mean']

## Initialize data loaders
trSet = roundDataset('data/TrainSet.mat')
valSet = roundDataset('data/ValSet.mat')

trDataloader = DataLoader(trSet,batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=trSet.collate_fn)
valDataloader = DataLoader(valSet,batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=valSet.collate_fn)

## Variables holding train and validation loss values:
train_loss = []
val_loss = []
prev_val_loss = math.inf

for epoch_num in range(pretrainEpochs+trainEpochs):

    if epoch_num == 0:
        print('Pre-training with MSE loss')
    elif epoch_num == pretrainEpochs:
        print('Training with NLL loss')


    ## Train:_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net.train_flag = True

    # Variables to track training performance:
    avg_tr_loss = 0
    avg_tr_time = 0
    avg_lat_acc = 0

    for i, data in enumerate(trDataloader):
        st_time = time.time()

        hist, nbrs, nbr_list_len, fut, lat_enc, op_mask, ds_ids, vehicle_ids, frame_ids, fut_anchred = data

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

        # Pre-train with MSE loss to speed up training
        if epoch_num < pretrainEpochs:
            fut_pred, _ = net(hist, nbrs, nbr_list_len, lat_enc)
            l = maskedMSE(fut_pred, fut_anchred, op_mask)
        else:
            fut_pred, lat_pred = net(hist, nbrs, nbr_list_len, lat_enc)
            # Train with NLL loss
            l = maskedNLL(fut_pred, fut_anchred, op_mask) + crossEnt(lat_pred, lat_enc)
            avg_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / \
                           lat_enc.size()[0]

        # Backprop and update weights
        optimizer.zero_grad()
        l.backward()
        a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()

        # Track average train loss and average train time:
        batch_time = time.time() - st_time
        avg_tr_loss += l.item()
        avg_tr_time += batch_time

        if i % 100 == 99:
            eta = avg_tr_time / 100 * (len(trSet) / batch_size - i)

            print("Epoch no:", epoch_num + 1,
                  "| Epoch progress(%):", format(i / (len(trSet) / batch_size) * 100, '0.2f'),
                  "| Avg train loss:", format(avg_tr_loss / 100, '0.4f'),
                  "| Acc:", format(avg_lat_acc, '0.4f'),
                  "| Validation loss prev epoch", format(prev_val_loss, '0.4f'),
                  "| ETA(s):", int(eta))

            train_loss.append(avg_tr_loss / 100)
            avg_tr_loss = 0
            avg_lat_acc = 0
            avg_tr_time = 0
    # _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

    # Validate:______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net.train_flag = False

    print("Epoch", epoch_num + 1, 'complete. Calculating validation loss...')
    avg_val_loss = 0
    avg_val_lat_acc = 0
    val_batch_count = 0
    total_points = 0

    for i, data in enumerate(valDataloader):
        st_time = time.time()
        # en_ex_enc
        hist, nbrs, nbr_list_len, fut, lat_enc, op_mask, ds_ids, vehicle_ids, frame_ids, fut_anchred = data

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

        if epoch_num < pretrainEpochs:
            # During pre-training with MSE loss, validate with MSE for true maneuver class trajectory
            net.train_flag = True
            fut_pred, _ = net(hist, nbrs, nbr_list_len, lat_enc)
            fut_pred = anchor_inverse(fut_pred, lat_enc, anchor_traj, args['d_s'], multi=False)
            l = maskedMSE(fut_pred, fut, op_mask)
        else:
            # During training with NLL loss, validate with NLL over multi-modal distribution

            fut_pred, lat_pred = net(hist, nbrs, nbr_list_len, lat_enc)

            fut_pred = anchor_inverse(fut_pred, lat_pred, anchor_traj, args['d_s'], multi=True)

            l = maskedNLLTest_LatInt(fut_pred, lat_pred, fut, op_mask, args['num_lat_classes'],
                                      use_maneuvers=True, avg_along_time=True)

            avg_val_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / \
                               lat_enc.size()[0]

        avg_val_loss += l.item()
        val_batch_count += 1

    print(avg_val_loss / val_batch_count)

    # Print validation loss and update display variables
    print('Validation loss :', format(avg_val_loss / val_batch_count, '0.4f'),
          "| Val Acc:", format(avg_val_lat_acc / val_batch_count * 100, '0.4f'))

    val_loss.append(avg_val_loss / val_batch_count)
    prev_val_loss = avg_val_loss / val_batch_count

# __________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

model_fname = 'trained_models/round_3D_Intention_4s_latOnly_anchor.tar'
torch.save(net.state_dict(), model_fname)




