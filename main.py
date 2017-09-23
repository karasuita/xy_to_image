import argparse
import os
import shutil
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import cv2

from model import Model
from dataset import ImageDataset, make_nz_tensor
from utils import load_img, save_img


DEFAULTS = {
    'image_file': 'images/original.png',
    'nz': 100,
    'nz2': 64,
}

TILE_SIZE = 64


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument('--image_file', type=str, default=DEFAULTS['image_file'])
    train_parser.add_argument('--output_dir', type=str, default='output')
    train_parser.add_argument('--nz', type=int, default=DEFAULTS['nz'])
    train_parser.add_argument('--nz2', type=int, default=DEFAULTS['nz2'])
    train_parser.add_argument('--batch_size', type=int, default=16)
    train_parser.add_argument('--epoch', type=int, default=300)
    train_parser.add_argument('--lr', type=float, default=1e-2)
    train_parser.add_argument('--beta1', type=float, default=0.5)
    train_parser.add_argument('--cuda', action='store_true')
    train_parser.add_argument('--video', action='store_true')
    train_parser.add_argument('--seed', type=int, default=710)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument('--image_file', type=str, default=DEFAULTS['image_file'])
    generate_parser.add_argument('--model_file', type=str, required=True)

    return parser.parse_args()


def create_logger(file_dir):
    from logging import getLogger, StreamHandler, FileHandler, DEBUG
    logger = getLogger(__name__)
    logger.setLevel(DEBUG)
    logger.propagate = False

    stream_handler = StreamHandler()
    logger.addHandler(stream_handler)

    file_path = os.path.join(file_dir, 'log.txt')
    file_handler = FileHandler(file_path)
    file_handler.setLevel(DEBUG)
    logger.addHandler(file_handler)

    return logger


def train(args, print_args=True):
    if print_args:
        print(args)

    cuda = args.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_dir = os.path.join(output_dir, datetime.now().strftime("%Y%m%d%H%M%S"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_dir = os.path.join(output_dir, 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_best_path = os.path.join( model_dir, 'model_best.zip')
    model_latest_path = os.path.join( model_dir, 'model_latest.zip')

    images_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    image = load_img(args.image_file)
    _, height, width = image.size()

    data_loader = DataLoader(dataset=ImageDataset(image, args.nz, TILE_SIZE), batch_size=args.batch_size, shuffle=True)
    model = Model(args.nz, args.nz2)
    criterion = nn.MSELoss()

    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    logger = create_logger(output_dir)

    best_epoch_loss = 1e20

    try:
        video = args.video
        video_writer = None
        if video:
            video_writer = cv2.VideoWriter(os.path.join(output_dir, 'video.avi'), 0, 5.0, (width, height))

        for epoch in range(1, args.epoch+1):
            epoch_loss = 0
            for batch in tqdm(iter(data_loader), 'Epoch {}'.format(epoch)):
                input, target = Variable(batch[0]), Variable(batch[1])
                if cuda:
                    input = input.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                loss = criterion(model(input), target)
                epoch_loss += loss.data[0]
                loss.backward()
                optimizer.step()

            # save model
            if cuda:
                model = model.cpu()

            model.save(model_latest_path)

            # save images
            img = generate_image(model, args.nz, height, width)
            save_img(img, os.path.join(images_dir, 'image_{:06}.png'.format(epoch - 1)))

            if video:
                img = img.numpy().transpose(1, 2, 0).clip(0, 1) * 255.0
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
                video_writer.write(img)

            if cuda:
                model = model.cuda()

            epoch_loss = epoch_loss / len(data_loader)
            if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                shutil.copy(model_latest_path, model_best_path)

            logger.info('Epoch {} ends({}): Ave. Loss: {}'.format(epoch, datetime.now().strftime('%Y/%m/%d %H:%M:%S'), epoch_loss))
    finally:
        if video_writer:
            video_writer.release()


def generate_image(model, nz, height, width):

    max_size = max(height, width)
    img = torch.FloatTensor(3, height, width)

    for y in tqdm(range(0, height, TILE_SIZE)):
        for x in range(0, width, TILE_SIZE):
            rate_x, rate_y = x / max_size, y / max_size
            X_input, Y_input = make_nz_tensor(nz, rate_x), make_nz_tensor(nz, rate_y)
            X = torch.cat((X_input, Y_input)).unsqueeze_(dim=0)
            X = Variable(X)
            Y = model(X).data[0]
            w = min(TILE_SIZE, width - x)
            h = min(TILE_SIZE, height - y)
            img[:,y:y+h,x:x+w] = Y[:,0:h,0:w]

    return img


def generate(args, print_args=True):
    if print_args:
        print(args)

    model = Model.load(args.model_file)

    image = load_img(args.image_file)
    _, height, width = image.size()
    max_size = max(height, width)

    img = generate_image(model, model.nz, height, width)

    save_img(img, 'generate.png')


def main():
    args = parse_args()

    if args.subcommand is None:
        print('Error: main.py train or generate!')
    elif args.subcommand == 'train':
        train(args)
    elif args.subcommand== 'generate':
        generate(args)


if __name__ == '__main__':
    main()