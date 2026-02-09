import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='RAM')

def str2bool(v):
    return v.lower() in ('true', '1')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# system params
system_arg = add_argument_group('imaging system Params')
system_arg.add_argument('--frequency', '-f', type=int, default=3, help='working frequency, unit: GHz')
system_arg.add_argument('--distance', '-dis', type=int, default=40, help='distance between RIS and ROI along x-axis, unit: wavelength')
system_arg.add_argument('--RIS_num2', type=str, default='50,50', help='RIS element number along y- and z-axis')
system_arg.add_argument('--ROI_num3', type=str, default='1,100,100', help='voxel numbers along x- y- z-axis')
system_arg.add_argument('--voxel_size_scale', type=float, default=5, help='lambda/voxel_size_scale')
system_arg.add_argument('--Tx', type=str, default='10,10,0', help='Tx position, unit: wavelength')
system_arg.add_argument('--Rx', type=str, default='10,-10,0', help='Rx position, unit: wavelength')
system_arg.add_argument('--antenna_num_tx', type=int, default=8, help='antenna number of the TX')
system_arg.add_argument('--antenna_num_rx', type=int, default=8, help='antenna number of the RX')
system_arg.add_argument('--antenna_direction', type=str, default='y', help='ULA, antenna arranged along which axis')
system_arg.add_argument('--num_glimpses', type=int, default=40, help='# certain number of glimpses')

# network params
net_arg = add_argument_group('Network Params')
net_arg.add_argument('--net_indim', type=int, default=512, help='input size for MLP, this is related to position encoding')
net_arg.add_argument('--net_outdim', type=int, default=1, help='output size for MLP, set for 1 since only output scattering coefficient')
net_arg.add_argument('--net_depth', type=int, default=6, help='depth of MLP layers')
net_arg.add_argument('--net_width', type=int, default=256, help='width of MLP layers')
net_arg.add_argument('--pe_indim', type=int, default=2, help='number of position values')
net_arg.add_argument('--pe_embedding_size', type=int, default=256, help='row of matrix B, pe_embedding_size*2 is MLP input dim')
net_arg.add_argument('--pe_value_scale', type=float, default=4.0, help='scale the values output by position encoding to avoid large/small values')

# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--img_file', type=str, default='./data/752_00150.png', help='use which image')
train_arg.add_argument('--epochs', type=int, default=5000, help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=1e-4, help='Initial learning rate value')
train_arg.add_argument('--weight_decay', type=float, default=1e-8, help='for adam and rms')
train_arg.add_argument('--gradient_clip', type=float, default=1.0, help='Initial learning rate value')
train_arg.add_argument('--lr_patience', type=int, default=50, help='Number of epochs to wait before reducing lr')
train_arg.add_argument('--train_patience', type=int, default=200, help='Number of epochs to wait before stopping train')
train_arg.add_argument('--sparse_loss_scale', type=float, default=0.01, help='sparse regularization')
train_arg.add_argument('--wandb_flag', type=str2bool, default=True, help='wandb project')
train_arg.add_argument('--wandb_project', type=str, default='INRImager', help='wandb project name')
train_arg.add_argument('--value_scale', type=float, default=110, help='scale CSI data to prevent small values')

# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--random_seed', type=int, default=2, help='Seed to ensure reproducibility')
misc_arg.add_argument('--ckpt_dir', type=str, default='./ckpt/', help='Directory in which to save model checkpoints')
misc_arg.add_argument('--logs_dir', type=str, default='./logs/', help='Directory in which Tensorboard logs wil be stored')
misc_arg.add_argument('--load', type=str2bool, default=False, help='Whether to resume training from checkpoint')
misc_arg.add_argument('--load_model_dir', type=str, default='', help='load model dir for continuing training')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
