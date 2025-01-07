import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

import torchvision.utils as vutils

import gymnasium as gym

from atari_batch import iterate_batches
from atari_model import Generator, Discriminator
from atari_wrapper import InputWrapper

log = gym.logger
log.set_level(gym.logger.INFO)

LATENT_VECTOR_SIZE = 100
DISCR_FILTERS = 64
GENER_FILTERS = 64
BATCH_SIZE = 16

# dimension input image will be rescaled
IMAGE_SIZE = 64

LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000
SAVE_MODEL_EVERY_ITER = 1000


device = torch.device('cuda')

envs = [InputWrapper(gym.make(name)) for name in ('Breakout-v4', 'AirRaid-v4', 'Pong-v4')]
shape = envs[0].observation_space.shape

net_discr = Discriminator(input_shape=shape).to(device)
net_gener = Generator(output_shape=shape).to(device)

# # Load the saved model weights 
# net_gener.load_state_dict(torch.load('Environment\SynthAtari\generator_10k.pth')) 
# net_discr.load_state_dict(torch.load('Environment\SynthAtari\discriminator_10k.pth'))

objective = nn.BCELoss()

gen_optimizer = optim.Adam(params=net_gener.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
dis_optimizer = optim.Adam(params=net_discr.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

writer = SummaryWriter()

gen_losses = []
dis_losses = []
iter_no = 0

true_labels_v = torch.ones(BATCH_SIZE, device=device)
fake_labels_v = torch.zeros(BATCH_SIZE, device=device)

ts_start = time.time()

for batch_v in iterate_batches(envs):
    # fake samples, input: batch, filters, x, y
    gen_input_v = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
    gen_input_v.normal_(0, 1)
    gen_input_v = gen_input_v.to(device)
    
    batch_v = batch_v.to(device)
    gen_output_v = net_gener(gen_input_v)

    # train discriminator
    dis_optimizer.zero_grad()
    dis_output_true_v = net_discr(batch_v)
    dis_output_fake_v = net_discr(gen_output_v.detach())
    
    dis_loss = objective(dis_output_true_v, true_labels_v) + objective(dis_output_fake_v, fake_labels_v)
    dis_loss.backward()
    dis_optimizer.step()
    
    dis_losses.append(dis_loss.item())

    # train generator
    gen_optimizer.zero_grad()
    dis_output_v = net_discr(gen_output_v)

    gen_loss_v = objective(dis_output_v, true_labels_v)
    gen_loss_v.backward()
    gen_optimizer.step()

    gen_losses.append(gen_loss_v.item())

    iter_no += 1
    if iter_no % REPORT_EVERY_ITER == 0:
        dt = time.time() - ts_start
        log.info("Iter %d in %.2fs: gen_loss=%.3e, dis_loss=%.3e", iter_no, dt, np.mean(gen_losses), np.mean(dis_losses))
        ts_start = time.time()
        
        writer.add_scalar("gen_loss", np.mean(gen_losses), iter_no)
        writer.add_scalar("dis_loss", np.mean(dis_losses), iter_no)
            
        gen_losses = []
        dis_losses = []

    if iter_no % SAVE_IMAGE_EVERY_ITER == 0:

        img = vutils.make_grid(gen_output_v.data[:64], normalize=True)
        writer.add_image("fake", img, iter_no)
        
        img = vutils.make_grid(batch_v.data[:64], normalize=True)
        writer.add_image("real", img, iter_no)

    if iter_no % SAVE_MODEL_EVERY_ITER == 0: 
        torch.save(net_gener.state_dict(), f'generator_{iter_no}.pth') 
        torch.save(net_discr.state_dict(), f'discriminator_{iter_no}.pth') 
        
envs.close()


