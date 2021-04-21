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

# Number of the lateral and longitudinal classes
args['num_lat_classes'] = 8
args['num_lon_classes'] = 3

# History and Future Horizons and Sampling rate
args['t_h'] = 50
args['t_f'] = 100
args['d_s'] = 4

# Number of training epochs
args['pretrainEpochs'] = 5
args['trainEpochs'] = 3

# Using  intention prediction and anchor trajectories
args['use_intention'] = True
args['use_anchors'] = True



