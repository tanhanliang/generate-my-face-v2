import dataset
import began
import torch
from torch.utils import data
import torch.optim as optim
from PIL import Image
import numpy as np
from torchvision.transforms import ToPILImage
from IPython.core.display import display
import math

device = torch.device('cuda:0')
params = {
    'batch_size': 16,
    'shuffle': True, 
    'num_workers': 4,
}
epochs = 300
learning_rate = .0001
img_size = 128
channels = 128
kernel_size = 3 # Even numbered kernel_size not supported
gamma = 0.5
lmbda = 0.01
dropout = 0.0
lr_decay = 0.9
lr_decay_epoch_multiple = 4
g_dropout = 0.5
d_dropout = 0.0
model = began.BEGAN(img_size, channels, kernel_size, gamma, lmbda, d_dropout, g_dropout)

print("Number of trainable parameters for discriminator/generator: ")
print(sum([param.nelement() for param in model.discriminator.parameters()]))

discrim_optimizer = optim.Adam(model.discriminator.parameters(), lr=learning_rate)
gen_optimizer = optim.Adam(model.generator.parameters(), lr=learning_rate)
model.set_optimizers(discrim_optimizer, gen_optimizer)

training_set = dataset.PictureDataset(img_size, 'data/', '.jpg')
training_generator = data.DataLoader(training_set, **params)

global_step = 0
num_images = training_generator.dataset.__len__()
batches = math.ceil(num_images/training_generator.batch_size)
batches = torch.tensor(batches).float().to(device)

conv_values = []

for epoch in range(121, epochs):
    total_conv_measure = torch.tensor(0.0).to(device)
    total_d_reconstr_loss = torch.tensor(0.0).to(device)
    total_g_reconstr_loss = torch.tensor(0.0).to(device)
    
    for batch_images in training_generator:
        batch_images = batch_images.to(device)

        model.train_batch(batch_images)
        global_step += 1
        
        total_conv_measure += model.batch_conv_measure
        total_d_reconstr_loss += model.batch_d_reconstr_loss
        total_g_reconstr_loss += model.batch_g_reconstr_loss
                
    print('%d[Epoch: %d][Conv: %1.4f][d_recon_loss: %1.4f][g_recon_loss: %1.4f][k: %f]' 
        % (
            global_step, 
            epoch, 
            total_conv_measure/batches, 
            total_d_reconstr_loss/batches, 
            total_g_reconstr_loss/batches, 
            model.k,
        ))
    conv_values.append((total_conv_measure/batches).item())

    # Learning rate decay
    if epoch % lr_decay_epoch_multiple == 0:
        learning_rate = learning_rate*lr_decay
        discrim_optimizer = optim.Adam(model.discriminator.parameters(), lr=learning_rate)
        gen_optimizer = optim.Adam(model.generator.parameters(), lr=learning_rate)
        model.set_optimizers(discrim_optimizer, gen_optimizer)
    
    noise = torch.rand(1, 3, img_size, img_size).to(device)
    fake_img = model.generator(noise)
    image = ToPILImage()(fake_img[0].detach().cpu())
    # print('Generated fake img')
    # display(image)
    image.save('outputs/generated-img%s.png' % str(epoch))

    r_fake = model.discriminator(fake_img)
    r_fake_image = ToPILImage()(r_fake[0].detach().cpu())
    # print('Reconstructed fake img')
    # display(r_fake_image)

    reconstr_real = model.discriminator(batch_images[0].unsqueeze(0))[0].detach().cpu()
    rr_image = ToPILImage()(reconstr_real)
    # print('Reconstructed real image')
    # display(rr_image)
    rr_image.save('reconstructed-real/recon-real%s.png' % str(epoch))
