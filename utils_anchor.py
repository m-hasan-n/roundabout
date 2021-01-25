
from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch

### Dataset class for the rounD dataset
class roundDataset(Dataset):

    def __init__(self, mat_file, t_h=50, t_f=100, d_s=4, enc_size=32, ip_dim=3, lat_dim=8, lon_dim=3, goal_dim = 16, en_ex_dim =2):

        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.A = scp.loadmat(mat_file)['anchor_traj_mean']

        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences
        self.enc_size = enc_size  # size of encoder LSTM
        self.ip_dim = ip_dim
        self.lat_dim = lat_dim


    def __len__(self):
        return len(self.D)


    def __getitem__(self, idx):
        # print('getitem is called ')
        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2]
        grid = self.D[idx,14:] #14 if no entry_exit_class 15 if there
        neighbors = []

        # Encoding of Lateral and Longitudinal Intention Classes
        lat_class = self.D[idx, 12] - 1
        lat_enc = np.zeros([self.lat_dim])
        lat_enc[int(lat_class )] = 1

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist = self.getHistory(vehId, t, vehId, dsId)
        fut, fut_anchred = self.getFuture(vehId, t, dsId, lat_class)

        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid:
            neighbors.append(self.getHistory(i.astype(int), t, vehId, dsId))



        return hist, fut, neighbors, lat_enc, dsId, vehId, t, fut_anchred

    ## Helper function to get track history
    def getHistory(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, self.ip_dim])
        else:
            veh_tracks = self.T

            if veh_tracks.shape[1] <= vehId - 1:
                return np.empty([0, self.ip_dim])
            refTrack = veh_tracks[dsId - 1][refVehId - 1].transpose()
            vehTrack = veh_tracks[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:self.ip_dim + 1]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, self.ip_dim])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:self.ip_dim + 1] - refPos

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, self.ip_dim])
            return hist

    ## Helper function to get track future
    def getFuture(self, vehId, t, dsId, lat_class):

        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:self.ip_dim + 1]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s, 1:self.ip_dim + 1] - refPos

        anchor_traj = self.A[int(lat_class),0]
        anchor_traj = anchor_traj[0:-1:self.d_s,:]

        fut_anchred = anchor_traj[0:len(fut),:]-fut


        return fut, fut_anchred

    ## Collate function for dataloader
    def collate_fn(self, samples):
        # Initialize neighbors and neighbors length batches:
        # nbr_batch_size = 0
        nbr_list_len = torch.zeros(len(samples), 1)
        for sample_id, (_, _, nbrs, _, _, _, _, _) in enumerate(samples):
            nbr_list_len[sample_id] = sum([len(nbrs[i]) != 0 for i in range(len(nbrs))])

        nbr_batch_size = int((sum(nbr_list_len)).item())
        maxlen = self.t_h // self.d_s + 1
        nbrs_batch = torch.zeros(maxlen, nbr_batch_size, self.ip_dim)

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen, len(samples), self.ip_dim)
        fut_batch = torch.zeros(self.t_f // self.d_s, len(samples), self.ip_dim)
        op_mask_batch = torch.zeros(self.t_f // self.d_s, len(samples), self.ip_dim)
        ds_ids_batch = torch.zeros(len(samples), 1)
        vehicle_ids_batch = torch.zeros(len(samples), 1)
        frame_ids_batch = torch.zeros(len(samples), 1)
        lat_enc_batch = torch.zeros(len(samples), self.lat_dim)
        fut_anchred_batch = torch.zeros(self.t_f // self.d_s, len(samples), self.ip_dim)

        count = 0
        # , en_ex_enc, fut_anch
        for sampleId, (hist, fut, nbrs, lat_enc, ds_ids, vehicle_ids, frame_ids, fut_anchored) in enumerate(samples):

            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            for k in range(self.ip_dim):
                hist_batch[0:len(hist), sampleId, k] = torch.from_numpy(hist[:, k])
                fut_batch[0:len(fut), sampleId, k] = torch.from_numpy(fut[:, k])
                fut_anchred_batch[0:len(fut), sampleId, k] = torch.from_numpy(fut_anchored[:, k])

            op_mask_batch[0:len(fut), sampleId, :] = 1
            ds_ids_batch[sampleId, :] = torch.tensor(ds_ids.astype(np.float64))
            vehicle_ids_batch[sampleId, :] = torch.tensor(vehicle_ids.astype(np.float64))
            frame_ids_batch[sampleId, :] = torch.tensor(frame_ids.astype(int).astype(np.float64))
            lat_enc_batch[sampleId, :] = torch.from_numpy(lat_enc)

            # Set up neighbor, neighbor sequence length, and mask batches:
            for id, nbr in enumerate(nbrs):
                if len(nbr) != 0:
                    for k in range(self.ip_dim):
                        nbrs_batch[0:len(nbr), count, k] = torch.from_numpy(nbr[:, k])
                    count += 1

        return hist_batch, nbrs_batch, nbr_list_len, fut_batch, lat_enc_batch, op_mask_batch, ds_ids_batch, vehicle_ids_batch, frame_ids_batch, fut_anchred_batch


def anchor_inverse(fut_pred, lat_pred, anchor_traj, d_s, multi):
    if multi:
        fut_adjusted=[]
        for l in range(len(fut_pred)):
            fut_adjusted.append(anchor_inverse_core(fut_pred[l], lat_pred, anchor_traj, d_s))
    else:
        fut_adjusted = anchor_inverse_core(fut_pred, lat_pred, anchor_traj, d_s)

    return fut_adjusted



def anchor_inverse_core(fut_pred, lat_pred, anchor_traj, d_s):
    fut_adjusted = fut_pred
    for k in range(lat_pred.shape[0]):
        lat_class = torch.argmax(lat_pred[k, :]).detach()
        # lat_class = lat_pred[k].nonzero().size()[0]
        anchor_tr = anchor_traj[lat_class, 0]
        anchor_tr = torch.from_numpy(anchor_tr[0:-1:d_s, :])
        anchor_tr = anchor_tr.cuda()
        fut_adjusted[:, k, 0:3] = anchor_tr - fut_pred[:, k, 0:3]
    return fut_adjusted


## Custom activation for output layer
# Simiilar to Graves work, "Generating Sequences With Recurrent Neural Networks"
# But using multinomial normal distribution with 3 variables
# X, Y, Theta
def outputActivation(x):
    if x.shape[2] == 9:
        muX = x[:,:,0:1]
        muY = x[:,:,1:2]
        muTh = x[:,:,2:3]
        sigX = x[:,:,3:4]
        sigY = x[:,:,4:5]
        sigTh = x[:,:,5:6]
        rhoXY = x[:,:,6:7]
        rhoYTh = x[:, :, 7:8]
        rhoXTh = x[:, :, 8:9]

        sigX = torch.exp(sigX)
        sigY = torch.exp(sigY)
        sigTh = torch.exp(sigTh)
        rhoXY = torch.tanh(rhoXY)
        rhoYTh = torch.tanh(rhoYTh)
        rhoXTh = torch.tanh(rhoXTh)
        out = torch.cat([muX, muY, muTh, sigX, sigY, sigTh, rhoXY, rhoYTh, rhoXTh], dim=2)

    elif x.shape[2] == 7:
        muX = x[:, :, 0:1]
        muY = x[:, :, 1:2]
        muTh = x[:, :, 2:3]
        sigX = x[:, :, 3:4]
        sigY = x[:, :, 4:5]
        sigTh = x[:, :, 5:6]
        rho = x[:, :, 6:7]
        sigX = torch.exp(sigX)
        sigY = torch.exp(sigY)
        sigTh = torch.exp(sigTh)
        rho = 0.4*torch.tanh(rho)

        out = torch.cat([muX, muY, muTh, sigX, sigY, sigTh, rho], dim=2)


    elif x.shape[2] == 5:
        muX = x[:, :, 0:1]
        muY = x[:, :, 1:2]
        sigX = x[:, :, 2:3]
        sigY = x[:, :, 3:4]
        rho = x[:, :, 4:5]
        sigX = torch.exp(sigX)
        sigY = torch.exp(sigY)
        rho = torch.tanh(rho)
        out = torch.cat([muX, muY, sigX, sigY, rho], dim=2)

    return out

## Batchwise NLL loss, uses mask for variable output lengths
def maskedNLL(y_pred, y_gt, mask):

    op_dim = y_pred.shape[2]

    if op_dim==5:
        acc = torch.zeros_like(mask)
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        sigX = y_pred[:, :, 2]
        sigY = y_pred[:, :, 3]
        rho = y_pred[:, :, 4]
        ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        # If we represent likelihood in feet^(-1):
        out = 0.5 * torch.pow(ohr, 2) * (
                    torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY,
                                                                                                2) - 2 * rho * torch.pow(
                sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379
        # If we represent likelihood in m^(-1):
        # out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
        acc[:, :, 0] = out
        acc[:, :, 1] = out
        acc = acc * mask
        lossVal = torch.sum(acc) / torch.sum(mask)

    elif op_dim==7:
        # FInd the NLL
        nll = compute_nll_mat_red(y_pred, y_gt)

        # nll_loss tensor filled with the loss value
        nll_loss = torch.zeros_like(mask)
        nll_loss[:, :, 0] = nll
        nll_loss[:, :, 1] = nll
        nll_loss[:, :, 2] = nll

        # mask the loss and find the mean value
        nll_loss = nll_loss * mask
        lossVal = torch.sum(nll_loss) / torch.sum(mask)

    elif op_dim==9:
        # If we represent likelihood in feet^(-1):

        # FInd the NLL
        # nll = compute_nll(y_pred, y_gt)
        nll = compute_nll_mat(y_pred, y_gt)

        # nll_loss tensor filled with the loss value
        nll_loss = torch.zeros_like(mask)
        nll_loss[:, :, 0] = nll
        nll_loss[:, :, 1] = nll
        nll_loss[:, :, 2] = nll

        #mask the loss and find the mean value
        nll_loss = nll_loss*mask
        lossVal = torch.sum(nll_loss)/torch.sum(mask)

    return lossVal



## NLL for sequence, outputs sequence of NLL values for each time-step, uses mask for variable output lengths, used for evaluation
def maskedNLLTest_LatInt(fut_pred, lat_pred, fut, op_mask, num_lat_classes,use_maneuvers = True, avg_along_time = False):
    if use_maneuvers:
        acc = torch.zeros(op_mask.shape[0],op_mask.shape[1],num_lat_classes).cuda()
        # acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], num_lat_classes)

        count = 0

        for l in range(num_lat_classes):
            wts = lat_pred[:,l]
            wts = wts.repeat(len(fut_pred[0]),1)
            y_pred = fut_pred[l]
            y_gt = fut

            # FInd the NLL
            out = compute_nll_mat_red(y_pred, y_gt)

            # If we represent likelihood in m^(-1):
            # out = -(0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160)
            acc[:, :, count] =  out + torch.log(wts.cpu())
            count+=1
        acc = -logsumexp(acc, dim = 2)
        acc = acc * op_mask[:,:,0]
        if avg_along_time:
            lossVal = torch.sum(acc) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc,dim=1)
            counts = torch.sum(op_mask[:,:,0],dim=1)
            return lossVal,counts
    else:
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1).cuda()
        y_pred = fut_pred
        y_gt = fut
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        sigX = y_pred[:, :, 2]
        sigY = y_pred[:, :, 3]
        rho = y_pred[:, :, 4]
        ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        # If we represent likelihood in feet^(-1):
        out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2 * rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379
        # If we represent likelihood in m^(-1):
        # out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
        acc[:, :, 0] = out
        acc = acc * op_mask[:, :, 0:1]
        if avg_along_time:
            lossVal = torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc[:,:,0], dim=1)
            counts = torch.sum(op_mask[:, :, 0], dim=1)
            return lossVal,counts




## NLL for sequence, outputs sequence of NLL values for each time-step, uses mask for variable output lengths, used for evaluation
def maskedNLLTest(fut_pred, fut, op_mask, avg_along_time = False):
    y_pred = fut_pred
    y_gt = fut

    # FInd the NLL
    # If we represent likelihood in feet^(-1):
    out = compute_nll_mat_red(y_pred, y_gt)

    # acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1).cuda()
    acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1)
    acc[:, :, 0] = out
    acc = acc * op_mask[:, :, 0:1]

    if avg_along_time:
        lossVal = torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
        return lossVal
    else:
        lossVal = torch.sum(acc[:,:,0], dim=1)
        counts = torch.sum(op_mask[:, :, 0], dim=1)
        return lossVal,counts


# Compute the NLL using the formula of Multivariate Gaussian distribution
#In matrix form
def compute_nll_mat_red(y_pred, y_gt):
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    muTh = y_pred[:, :, 2]
    sigX = y_pred[:, :, 3]
    sigY = y_pred[:, :, 4]
    sigTh = y_pred[:, :, 5]
    rho = y_pred[:, :, 6]




    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    th = y_gt[:, :, 2]

    # XU = ([x - muX, y - muY, th - muTh])
    # XU = torch.cat((x - muX, y - muY, th - muTh),0)
    XU = torch.zeros(x.shape[0], x.shape[1], 3, 1)
    XU[:, :, 0, 0] = x - muX
    XU[:, :, 1, 0] = y - muY
    XU[:, :, 2, 0] = th - muTh

    #sigma
    sigma_mat = torch.zeros(x.shape[0], x.shape[1], 3, 3)
    sigma_mat[:, :, 0, 0] = torch.pow(sigX, 2)
    sigma_mat[:, :, 1, 0] = rho * sigX * sigY
    sigma_mat[:, :, 2, 0] = rho * sigX * sigTh

    sigma_mat[:, :, 0, 1] = rho * sigX * sigY
    sigma_mat[:, :, 1, 1] = torch.pow(sigY, 2)
    sigma_mat[:, :, 2, 1] = rho * sigY * sigTh

    sigma_mat[:, :, 0, 2] = rho * sigX * sigTh
    sigma_mat[:, :, 1, 2] = rho * sigY * sigTh
    sigma_mat[:, :, 2, 2] = torch.pow(sigTh, 2)

    loss_1 = 0.5 * torch.matmul(torch.matmul(XU.transpose(2, 3), sigma_mat.inverse()), XU)
    loss_1 = loss_1.view(x.shape[0], x.shape[1])



    nll_loss = loss_1 + 2.7568 + 0.5*torch.log(sigma_mat.det())

    # if use_reg:
    #     # rho_reg_term = 1 - 3 * torch.pow(rho, 2) + 2 * torch.pow(rho, 3)
    #     rho_reg_term = 3 * torch.pow(rho, 2) - 2 * torch.pow(rho, 3)
    #     # nll_loss = nll_loss + torch.pow(rho_reg_term.cpu(),2)
    #     nll_loss = nll_loss + torch.abs(rho_reg_term.cpu())

    return nll_loss

# Compute the NLL using the formula of Multivariate Gaussian distribution
#In matrix form
def compute_nll_mat(y_pred, y_gt):
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    muTh = y_pred[:, :, 2]
    sigX = y_pred[:, :, 3]
    sigY = y_pred[:, :, 4]
    sigTh = y_pred[:, :, 5]
    rhoXY = y_pred[:, :, 6]
    rhoYTh = y_pred[:, :, 7]
    rhoXTh = y_pred[:, :, 8]

    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    th = y_gt[:, :, 2]

    # XU = ([x - muX, y - muY, th - muTh])
    # XU = torch.cat((x - muX, y - muY, th - muTh),0)
    XU = torch.zeros(x.shape[0], x.shape[1], 3, 1)
    XU[:, :, 0, 0] = x - muX
    XU[:, :, 1, 0] = y - muY
    XU[:, :, 2, 0] = th - muTh

    #sigma
    sigma_mat = torch.zeros(x.shape[0], x.shape[1], 3, 3)
    sigma_mat[:, :, 0, 0] = torch.pow(sigX, 2)
    sigma_mat[:, :, 1, 0] = rhoXY * sigX * sigY
    sigma_mat[:, :, 2, 0] = rhoXTh * sigX * sigTh

    sigma_mat[:, :, 0, 1] = rhoXY * sigX * sigY
    sigma_mat[:, :, 1, 1] = torch.pow(sigY, 2)
    sigma_mat[:, :, 2, 1] = rhoYTh * sigY * sigTh

    sigma_mat[:, :, 0, 2] = rhoXTh * sigX * sigTh
    sigma_mat[:, :, 1, 2] = rhoYTh * sigY * sigTh
    sigma_mat[:, :, 2, 2] = torch.pow(sigTh, 2)

    loss_1 = 0.5 * torch.matmul(torch.matmul(XU.transpose(2, 3), sigma_mat.inverse()), XU)
    loss_1 = loss_1.view(x.shape[0], x.shape[1])

    nll_loss = loss_1 + 0.5*torch.log(sigma_mat.det()) + 2.7568

    return nll_loss


# Compute the NLL using the formula of Multivariate Gaussian distribution
def compute_nll(y_pred, y_gt):
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    muTh = y_pred[:, :, 2]
    sigX = y_pred[:, :, 3]
    sigY = y_pred[:, :, 4]
    sigTh = y_pred[:, :, 5]
    rhoXY = y_pred[:, :, 6]
    rhoYTh = y_pred[:, :, 7]
    rhoXTh = y_pred[:, :, 8]

    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    th = y_gt[:, :, 2]

    # If we represent likelihood in feet^(-1):
    eps_loss = 1e-6
    rho_f = 1 - torch.pow(rhoXY, 2) - torch.pow(rhoXTh, 2) - torch.pow(rhoYTh, 2) + 2 * rhoXY * rhoXTh * rhoYTh
    #=0 to avoid divide by 0 and <0 to avoid log(negative) when computing log(sig_det)
    rho_f[rho_f.clone() == 0] = eps_loss

    sig_det = torch.pow(sigX, 2) * torch.pow(sigY, 2) * torch.pow(sigTh, 2) * rho_f.clone()

    # XU = ([x - muX, y - muY, th - muTh])
    xu = x - muX
    yu = y - muY
    tu = th - muTh

    s11_d = torch.pow(sigX, 2)
    s11_d[s11_d.clone()==0]=eps_loss
    s11 = (1 - torch.pow(rhoYTh, 2)) / s11_d

    s12_d = (sigX * sigY)
    s12_d[s12_d.clone() == 0] = eps_loss
    s12 = (rhoXTh * rhoYTh - rhoXY) / s12_d

    s13_d = (sigX * sigTh)
    s13_d[s13_d.clone() == 0] = eps_loss
    s13 = (rhoXY * rhoYTh - rhoXTh) / s13_d

    s21 = s12

    s22_d = torch.pow(sigY, 2)
    s22_d[s22_d.clone() == 0] = eps_loss
    s22 = (1 - torch.pow(rhoXTh, 2)) / s22_d

    s23_d = (sigY * sigTh)
    s23_d[s23_d.clone() == 0] = eps_loss
    s23 = (rhoXY * rhoXTh - rhoYTh) / s23_d

    s31 = s13
    s32 = s23

    s33_d = torch.pow(sigTh, 2)
    s33_d[s33_d.clone() == 0] = eps_loss
    s33 = (1 - torch.pow(rhoXY, 2)) / s33_d

    s1 = xu * (xu*s11 + yu*s21 + tu*s31)
    s2 = yu * (xu*s12 + yu*s22 + tu*s32)
    s3 = tu * (xu*s13 + yu*s23 + tu*s33)

    nll_loss = 0.5*(s1+s2+s3)/rho_f + 0.5*torch.log(sig_det) + 2.7568

    return nll_loss


## Batchwise MSE loss, uses mask for variable output lengths
def maskedMSE(y_pred, y_gt, mask, use_reg=None):

    acc = torch.zeros_like(mask)

    ip_dim = y_gt.shape[2]

    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)

    if ip_dim==3:
        muTh = y_pred[:,:,2]
        th = y_gt[:,:, 2]
        out = out + torch.pow(th-muTh, 2)

    if use_reg:
        op_dim = y_pred.shape[2]
        if op_dim==5:
            rho = y_pred[:, :, 4]
        elif op_dim==7:
            rho = y_pred[:, :, 6]
        # rho_reg_term = 1 - 3 * torch.pow(rho, 2) + 2 * torch.pow(rho, 3)
        rho_reg_term = 3 * torch.pow(rho, 2) - 2 * torch.pow(rho, 3)
        # out = out + torch.pow(rho_reg_term,2)
        out = out + torch.abs(rho_reg_term)

    for k in range(ip_dim):
        acc[:, :, k] = out

    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal

## MSE loss for complete sequence, outputs a sequence of MSE values, uses mask for variable output lengths, used for evaluation
def maskedMSETest(y_pred, y_gt, mask):

    acc = torch.zeros_like(mask)

    ip_dim = y_gt.shape[2]

    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]

    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]

    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)

    if ip_dim==3:
        muTh = y_pred[:, :, 2]
        th = y_gt[:, :, 2]
        out = out + torch.pow(th-muTh, 2)

    for k in range(ip_dim):
        acc[:, :, k] = out

    acc = acc * mask
    lossVal = torch.sum(acc[:,:,0],dim=1)
    counts = torch.sum(mask[:,:,0],dim=1)
    return lossVal, counts

## Hasan: my function to save the evaluation results to csv files
def saveResultFiles(y_pred, y_gt,op_mask, results_file):
    for i in range(y_pred.size()[1]):
        muX = y_pred[:, i, 0]
        muY = y_pred[:, i, 1]

        x = y_gt[:, i, 0]
        y = y_gt[:, i, 1]

        xy_pred = np.column_stack((muX.detach().cpu().numpy(), muY.detach().cpu().numpy()))
        xy_gt = np.column_stack((x.detach().cpu().numpy(), y.detach().cpu().numpy()))

        mask = op_mask[:, i, 0]
        mask = mask.detach().cpu().numpy()
        xy = np.column_stack((xy_pred, xy_gt, mask))

        np.savetxt(results_file, xy)


## Hasan: my function to evaluate X and Y separately as wells as whole XY
def maskedMSETest_XY(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    acc_x = torch.zeros_like(mask)
    acc_y = torch.zeros_like(mask)

    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]

    out_x = torch.pow(x - muX, 2)
    out_y = torch.pow(y - muY, 2)
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)

    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask

    acc_x[:, :, 0] = out_x
    acc_x[:, :, 1] = out_x
    acc_x = acc_x * mask

    acc_y[:, :, 0] = out_y
    acc_y[:, :, 1] = out_y
    acc_y = acc_y * mask

    lossVal = torch.sum(acc[:,:,0],dim=1)
    lossVal_x = torch.sum(acc_x[:, :, 0], dim=1)
    lossVal_y = torch.sum(acc_y[:, :, 0], dim=1)

    counts = torch.sum(mask[:,:,0],dim=1)
    return lossVal, lossVal_x, lossVal_y, counts


## Helper function for log sum exp calculation:
def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs