import os
import cv2
import torch
import wandb
import psutil
import logging
import datetime

from torch import optim
from model import SIREN, PositionalEncoder, PhysicalModel

from config import get_config
from channel_generation import channel_generation
from skimage.metrics import structural_similarity as ssim
from utils import seed_everything, prepare_dirs, save_config, create_grid, save_checkpoint

os.environ["WANDB_SILENT"] = "true"


def train_model(
        model,
        optimizer,
        encoder,
        config,
        true_img,
        used_channel,
        supervise_data,
):

    img_dim = true_img.shape
    grid = create_grid(img_dim[0], img_dim[1]).unsqueeze(dim=0).to(device=config.device)
    embeds = encoder.embedding(grid)
    Physical = PhysicalModel(used_channel, config.value_scale)
    # when training NN, additive noise at the RX is unknown, but used subchannel is inaccurate
    
    # loss function
    loss1 = torch.nn.MSELoss()
    loss2 = torch.nn.L1Loss()
    zz = torch.zeros(true_img.shape).to(device=config.device)
    best_train_mse = 1e8
    counter_lr = 0
    counter_terminal = 0
    best_epoch = 0

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()

        train_out = model(embeds) # model predicts pixel values for all positions
        train_out_project = Physical.illuminate(train_out.view(1, -1))
        # calculate the channel response corresponding to the image

        train_mse = loss1(train_out_project, supervise_data)
        train_loss = train_mse + config.sparse_loss_scale * loss2(train_out.squeeze(), zz) # calculate loss
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip) # gradient clip
        optimizer.step()

        # record training data
        train_psnr = (-10 * train_mse.log10()).detach().cpu().numpy()
        train_mse = train_mse.item()
        train_loss = train_loss.item()

        # test using the true image
        test_mse = loss1(train_out.squeeze(), true_img)
        test_psnr = - 10 * torch.log10(test_mse).item()
        test_mse = test_mse.item()
        test_ssim, diff = ssim(train_out.squeeze().detach().cpu().numpy(), true_img.detach().cpu().numpy(), data_range=1.0, full=True)

        # record if this is the best model so far using the training data
        is_best = train_mse < best_train_mse
        if is_best:
            counter_lr = 0
            counter_terminal = 0
            best_epoch = epoch
            best_train_mse = train_mse
            test_mse_best_model = test_mse
            test_psnr_best_model = test_psnr
            test_ssim_best_model = test_ssim
        else:
            counter_lr += 1
            counter_terminal += 1

        # logging training information
        msg = 'epoch: {} - train loss: {:.4f} - train mse: {:.4f} - train psnr: {:.4f} - test mse: {:.4f} - test psnr: {:.4f} - test ssim: {:.4f}'
        if is_best: msg = msg + ' [*]'
        logging.info(msg.format(epoch, train_loss, train_mse, train_psnr, test_mse, test_psnr, test_ssim))
        
        # wandb output and checkpoint save
        if config.wandb_flag:
            experiment.log({'epoch': epoch, 'learning rate': optimizer.param_groups[0]['lr'],
                            'train loss': train_loss, 'train mse': train_mse, 'train psnr': train_psnr,
                            'test mse': test_mse, 'test psnr': test_psnr, 'test ssim': test_ssim,
                            'test mse best model': test_mse_best_model, 'test psnr best model': test_psnr_best_model, 
                            'test ssim best model': test_ssim_best_model, 'no improvement': counter_terminal, 'best epoch': best_epoch})
            if epoch % 50 == 0: # upload gradients, weights, and images every 50 epochs
                histograms = {}
                for tag, value in model.named_parameters():
                    tag = tag.replace('/', '.')
                    if not (torch.isinf(value) | torch.isnan(value)).any(): histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                    if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any(): histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                experiment.log({**histograms, 'predict image': wandb.Image(train_out.cpu().squeeze().data), 'diff image': wandb.Image(diff)})

            # save checkpoint only when using wandb
            if counter_terminal > config.train_patience:
                # terminal when no improvements for config.train_patience epochs
                logging.info("[!] No improvement in {} epochs, stopping training.".format(config.train_patience))
                save_checkpoint(epoch, model.state_dict(), optimizer.state_dict(), config.ckpt_dir, encoder.B, 1)
                return # this is unable for CPU running. no influences since CPU is used for debugging, and all formal training is on GPU
            if epoch == config.epochs - 1:
                # terminal when reaching the last epoch
                logging.info("[!] Reaching the last epoch")
                save_checkpoint(epoch, model.state_dict(), optimizer.state_dict(), config.ckpt_dir, encoder.B, 1)
            if is_best:
                # save checkpoint for the best model
                save_checkpoint(epoch, model.state_dict(), optimizer.state_dict(), config.ckpt_dir, encoder.B, 2)

        if train_mse > best_train_mse * 10: # this is to handle the scenario that sudden bad mse accurs (unstability training)
            checkpoint = torch.load(os.path.join(config.ckpt_dir, 'model_best.pth'))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for param_group in optimizer.param_groups: param_group["lr"] *= 0.5
            counter_lr = 0
            logging.info('train_mse: {:.4f} is larger than 10 times of the saved best mse: {:.4f}'.format(train_mse, best_train_mse))
            logging.info('saved best model at epoch {} is loaded, and the learning rate is descended by half'.format(checkpoint['epoch']))
        if counter_lr > config.lr_patience: # do not use scheduler, but manually cut down lr
            for param_group in optimizer.param_groups: param_group["lr"] *= 0.5
            counter_lr = 0 # since manually reduce lr, another counter is required for terminal training
            logging.info('train_mse is not imporved for config.lr_patience: {} epochs, the learning rate is descended by half'.format(config.lr_patience))


if __name__ == '__main__':
    config, unparsed = get_config() # get configs, set random seed, examine input
    seed_everything(config.random_seed)

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.device.type == 'cpu': config.num_workers = 1
    config.running_code = os.path.abspath(__file__) # record running which code

    # initial wandb
    if config.wandb_flag:
        experiment = wandb.init(project=config.wandb_project, resume='allow', anonymous='must', config=config)
        config.time_str_now = os.path.relpath(wandb.run.dir, start=os.getcwd())[10:25] + '_wandbID_' + wandb.run.name.split('-')[-1] # get present time and wandb ID
        config.ckpt_dir = config.ckpt_dir + config.time_str_now
        prepare_dirs(config) # create log and ckpt dirs
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', filename=config.logs_dir + config.time_str_now + '.txt') # logs saved to a .txt file
        save_config(config, os.path.join(config.logs_dir, config.time_str_now + '_params.json')) # save config
        experiment.config.update(config, allow_val_change=True) # update config to wandb
    else:
        config.time_str_now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s') # logs presented at the terminal
    logging.info('CMD line:\n{}\n'.format(' '.join(psutil.Process().cmdline()))) # record CMD line
    logging.info(f'Using device: {config.device}, log time: {config.time_str_now}, using wandb: {config.wandb_flag}, wandb project name: {config.wandb_project}')

    # calculate necessary data
    used_channel = channel_generation(config)
    used_channel = {key: tensor.to(config.device) for key, tensor in used_channel.items()} # convert channels to GPU
    true_img = cv2.imread(config.img_file, cv2.IMREAD_GRAYSCALE) / 255
    ROI_num3 = list(map(lambda x: int(x), config.ROI_num3.split(',')))
    if not (ROI_num3[1] == true_img.shape[0] and ROI_num3[2] == true_img.shape[1]):
        true_img = cv2.resize(true_img, (ROI_num3[1], ROI_num3[2]))
    true_img = torch.tensor(true_img, dtype=torch.float32).to(device=config.device)
    img_vector = true_img.view(1, -1)

    Physical = PhysicalModel(used_channel, config.value_scale)
    supervise_data = Physical.illuminate(img_vector)

    # initial model, optimizer, and PositionalEncoder
    model = SIREN(config)
    logging.info({model})
    optimizer = optim.Adam(model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay)
    encoder = PositionalEncoder(config)

    # load data and move to GPU
    if config.load:
        para_dict = torch.load(config.load_model_dir, map_location=config.device)
        model.load_state_dict(para_dict['model_state_dict'])
        optimizer.load_state_dict(para_dict['optimizer_state_dict'])
        encoder.B = para_dict['B']
        logging.info(f'Model loaded from {config.load_model_dir}')

    model.to(device=config.device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    train_model(
        model=model,
        optimizer=optimizer,
        encoder=encoder,
        config=config,
        true_img=true_img,
        used_channel=used_channel,
        supervise_data=supervise_data,
    )

