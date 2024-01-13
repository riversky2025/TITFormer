import os
import pickle
import logging
import sys

import kornia
import torch
import numpy as np

from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import utils
from model import TITFormer, Enhancer

from utils import create_dir, logger, set_random_seed, CharbonnierLoss, create_folder_for_run
from collections import defaultdict
from registry import DATASET_REGISTRY
import dataset


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val != np.nan and val != np.inf:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


from omegaconf import OmegaConf
def train():
    args = OmegaConf.load("option/trainxyz.yaml")
    print(args.runsetting.filepath)
    trainOpt = args.train
    dataSetOpt = args.dataset
    checkpoint = None
    loaded_checkpoint_file_name = None
    if args.runsetting.command == 'new':
        runName = "{}-{}-{}".format(dataSetOpt.name[:3], dataSetOpt.name[-3:], dataSetOpt.version[-1:])
        trainOpt.epoch = 0

        this_run_folder = create_folder_for_run("./run", runName)
        logging.basicConfig(level=logging.INFO,
                            format='%(message)s',
                            handlers=[
                                logging.FileHandler(
                                    os.path.join(this_run_folder, f'{runName}.log')),
                                logging.StreamHandler(sys.stdout)
                            ])

        with open(os.path.join(this_run_folder, 'options-and-config.pickle'), 'wb+') as f:
            pickle.dump(trainOpt, f)
            pickle.dump(dataSetOpt, f)
    else:

        this_run_folder = args.runsetting.filepath
        options_file = os.path.join(this_run_folder, 'options-and-config.pickle')
        trainOpt, dataSetOpt = utils.load_options(options_file)
        trainOpt.device=args.runsetting.device
        runName = "{}-{}-{}".format(dataSetOpt.name[:3], dataSetOpt.name[-3:], dataSetOpt.version[-1:])

        checkpoint, loaded_checkpoint_file_name = utils.load_last_checkpoint(
            os.path.join(this_run_folder, 'checkpoints'))
        trainOpt.epoch = checkpoint['epoch'] + 1
        if args.train.n_epochs is not None:
            if trainOpt.epoch >= args.train.n_epochs:
                print(
                    f'Command-line specifies of number of epochs = {args.train.n_epochs}, but folder={this_run_folder} '
                    f'already contains checkpoint for epoch = {trainOpt.epoch}.')
                exit(1)
        if args.dataset.data_root is not None:
            dataSetOpt.data_root = args.dataset.data_root

    set_random_seed(trainOpt.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    enhancer = Enhancer(trainOpt)
    if args.runsetting.command == 'continue':
        # if we are continuing, we have to load the model params
        assert checkpoint is not None
        logging.info(f'Loading checkpoint from file {loaded_checkpoint_file_name}')
        utils.model_from_checkpoint(enhancer, checkpoint)
    logging.info('Begin Training......')

    train_data = DataLoader(
        DATASET_REGISTRY.get(dataSetOpt.name)(dataSetOpt, mode="train"),
        batch_size=dataSetOpt.batch_size,
        shuffle=True,
        num_workers=dataSetOpt.n_cpu,
    )

    val_data = DataLoader(
        DATASET_REGISTRY.get(dataSetOpt.name)(dataSetOpt, mode="test"),
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    logging.info('Number of training images: {:d}'.format(len(train_data)))
    logging.info('Number of validation images: {:d}'.format(len(val_data)))
    for epoch in range(trainOpt.epoch, trainOpt.n_epochs):
        training_meter = defaultdict(AverageMeter)
        loop = tqdm(train_data, total=len(train_data))
        enhancer.model.train()
        for i, batch in enumerate(loop):
            meters, _ = enhancer.train_on_batch(batch)
            for name, meter in meters.items():
                training_meter[name].update(meter)
            if i % trainOpt.print_each_step == 0:
                loop.set_description(
                    "{}t|E:{}/{} loss:{:.4f},psnr:{:.4f}"
                        .format(trainOpt.device,epoch, trainOpt.n_epochs, training_meter['loss'].avg, training_meter['psnr'].avg))
            # logging.info("Iteration: {:0>3}, Loss: {:.8f}".format(curr_itrs, loss.item()))
        enhancer.sche_step()
        utils.write_losses(os.path.join(this_run_folder, 'train.csv'), training_meter, epoch)
        first_iteration = True
        val_average = defaultdict(AverageMeter)
        loopVal = tqdm(val_data, total=len(val_data))
        enhancer.model.eval()
        saved_images_size = (512, 512)
        for i, batch in enumerate(loopVal):

            meters, (inputs, labels, outputs, inf, text) = enhancer.validate_on_batch(batch)
            for name, meter in meters.items():
                val_average[name].update(meter)
            if first_iteration:
                utils.save_images(inputs.cpu()[:1, :, :, :],
                                  inf[:1, :, :, :].cpu(),
                                  outputs[:1, :, :, :].cpu(),
                                  labels[:1, :, :, :].cpu(),
                                  epoch,
                                  os.path.join(this_run_folder, 'images'), resize_to=saved_images_size)
                first_iteration = False
            if i % trainOpt.print_each_step == 0:
                loopVal.set_description(
                    "v|E:{}/{} loss:{:.4f},psnr:{:.4f}"
                        .format(epoch, trainOpt.n_epochs,
                                val_average['loss'].avg, val_average['psnr'].avg))

        utils.log_progress(val_average)
        logging.info('-' * 40)
        utils.save_checkpoint(enhancer, runName, epoch, os.path.join(this_run_folder, 'checkpoints'))
        utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), val_average, epoch)

    logging.info('End of the training.')


if __name__ == '__main__':
    train()
