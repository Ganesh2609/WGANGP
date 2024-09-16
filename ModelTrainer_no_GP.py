import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_generator(generator:torch.nn.Module,
                    discriminator:torch.nn.Module,
                    optimizer:torch.optim.Optimizer,
                    BATCH_SIZE:int,
                    Z_DIM:int,
                    device:torch.device):
    
    generator.train()
    discriminator.eval()
    
    noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)
    fake_img = generator(noise)
    pred_probs = discriminator(fake_img)
    loss = -torch.mean(pred_probs)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    del noise, fake_img, pred_probs
    return loss.item()


def train_discriminator(generator:torch.nn.Module,
                        discriminator:torch.nn.Module,
                        batch:tuple,
                        optimizer:torch.optim.Optimizer,
                        BATCH_SIZE:int,
                        Z_DIM:int,
                        WEIGHT_CLIP:int,
                        device:torch.device):
    
    generator.eval()
    discriminator.train()
    
    real_img, _ = batch
    real_img = real_img.to(device)
      
    noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)
    fake_img = generator(noise)
    
    real_probs = discriminator(real_img)
    fake_probs = discriminator(fake_img)
    real_score = torch.mean(real_probs)
    fake_score = torch.mean(fake_probs)
    loss = fake_score - real_score
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        
    for p in discriminator.parameters():
        p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)
        
    del real_img, fake_img, real_probs, fake_probs
    return loss.item(), real_score.item(), fake_score.item()


def train_models(generator:torch.nn.Module,
                 discriminator:torch.nn.Module,
                 dataloader:torch.utils.data.DataLoader,
                 gen_optimizer:torch.optim.Optimizer,
                 disc_optimizer:torch.optim.Optimizer,
                 BATCH_SIZE:int,
                 Z_DIM:int,
                 NUM_EPOCHS:int,
                 device:torch.device,
                 DISC_ITERATIONS:int=5,
                 WEIGHT_CLIP:int=0.01,
                 gen_path:str=None,
                 disc_path:str=None,
                 result_path:str=None):
    
    results = {
        'Generator Loss' : [],
        'Discriminator Loss' : []
    }
    
    
    for epoch in range(1, NUM_EPOCHS+1):

        disc_loss = 0 
        gen_loss = 0   
        with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
            
            ITER_COUNT = DISC_ITERATIONS
            disc_batch_loss, real_score, fake_score = 1, 1, 1
            
            for i, batch in t:
                
                if ITER_COUNT < DISC_ITERATIONS:
                    disc_batch_loss, real_score, fake_score = train_discriminator(generator=generator, 
                                                                                discriminator=discriminator, 
                                                                                batch=batch, 
                                                                                optimizer=disc_optimizer, 
                                                                                BATCH_SIZE=BATCH_SIZE, 
                                                                                Z_DIM=Z_DIM,
                                                                                WEIGHT_CLIP=WEIGHT_CLIP,
                                                                                device=device)
                    ITER_COUNT+=1
                    continue
                ITER_COUNT = 0
                
                gen_batch_loss = train_generator(generator=generator,
                                                 discriminator=discriminator,
                                                 optimizer=gen_optimizer,
                                                 BATCH_SIZE=BATCH_SIZE,
                                                 Z_DIM=Z_DIM,
                                                 device=device)
                
                disc_loss += disc_batch_loss
                gen_loss += gen_batch_loss

                t.set_description(f'Epoch [{epoch}/{NUM_EPOCHS}] ')
                t.set_postfix({
                    'Gen Batch Loss' : gen_batch_loss,
                    'Gen Loss' : gen_loss/((i+1)//5),
                    'Disc Batch Loss' : disc_batch_loss,
                    'Disc Loss' : disc_loss/((4*(i+1))//5),
                    'Real' : real_score,
                    'Fake' : fake_score
                })
                
                if gen_path:
                    torch.save(obj=generator.state_dict(), f=gen_path)
                if disc_path:
                    torch.save(obj=discriminator.state_dict(), f=disc_path)
                
                # Save results every 100 batches
                if i % 100 == 0 and result_path:
                    RESULT_SAVE_NAME = result_path + f'/Epoch_{epoch}.png'
                    generator.eval()
                    discriminator.eval()
                    with torch.inference_mode():
                        noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)
                        fake_img = generator(noise).cpu()
                    fake_img = torch.clamp((fake_img + 1) / 2, 0, 1)
                    
                    fig, ax = plt.subplots(4, 8, figsize=(15, 8))
                    for i, ax in enumerate(ax.flat):
                        ax.imshow(fake_img[i].permute(1,2,0))
                        ax.axis(False);
                    plt.tight_layout()
                    plt.savefig(RESULT_SAVE_NAME)
                    plt.close(fig)
                
        results['Generator Loss'].append(gen_loss/len(dataloader))
        results['Discriminator Loss'].append(disc_loss/len(dataloader))
        
    return results