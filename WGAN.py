import torch 
from torch import nn 


class Discriminator(nn.Module):
    
    def __init__(self, in_channels:int, hidden_channels:int):
        super(Discriminator, self).__init__()
        self.dis_net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            self.conv_block(in_channels=hidden_channels, out_channels=hidden_channels*2, kernel_size=4, stride=2, padding=1),
            self.conv_block(in_channels=hidden_channels*2, out_channels=hidden_channels*4, kernel_size=4, stride=2, padding=1),
            self.conv_block(in_channels=hidden_channels*4, out_channels=hidden_channels*8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_channels=hidden_channels*8, out_channels=1, kernel_size=4, stride=2, padding=0),
            nn.Flatten()
        )
        
    def conv_block(self, in_channels:int, out_channels:int, kernel_size:int, stride:int, padding:int):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        return self.dis_net(x)
    

class Generator(nn.Module):
    
    def __init__(self, latent_channels:int, hidden_channels:int, img_channels:int):
        super(Generator, self).__init__()
        self.gen_net = nn.Sequential(
            self.conv_block(in_channels=latent_channels, out_channels=hidden_channels*16, kernel_size=4, stride=1, padding=0),
            self.conv_block(in_channels=hidden_channels*16, out_channels=hidden_channels*8, kernel_size=4, stride=2, padding=1),
            self.conv_block(in_channels=hidden_channels*8, out_channels=hidden_channels*4, kernel_size=4, stride=2, padding=1),
            self.conv_block(in_channels=hidden_channels*4, out_channels=hidden_channels*2, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=hidden_channels*2, out_channels=img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def conv_block(self, in_channels:int, out_channels:int, kernel_size:int, stride:int, padding:int):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.gen_net(x)
    
    
def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)