import torch
import torch.nn.functional as F

"""
So basically we don't want to add the generator loss from the start because 
1) The discriminator hasn't learned anything yet, so it's not going to sent a reliable signal
2) The generator is not doing well, so reconstruction/perceptual loss should dominate anyways

the unfortunate thing is that vqgan has the generator loss start kicking in at 250001 steps, and I'm
only training for 250000 steps in the first place, for now. This means I need to either train for longer
or activate the generator loss sooner. Maybe I'll implement something that starts the generator loss once 
discriminator loss variance drops below a certain point over the last 10 epochs?
"""
def adopt_generator_weight(weight, global_step, threshold = 0, value = 0):
    if global_step < threshold:
        weight = value
    return weight

def calculate_d_loss(logits_real, logits_fake):
    # real should be positive fake should be negative
    loss_real = torch.mean(F.softplus(-logits_real)) # --> 0 when logits are positive --> inf when logits are negative
    loss_fake = torch.mean(F.softplus(logits_fake)) # --> 0 when logits are negative --> inf when logits are positive
    return (loss_real + loss_fake) / 2

# If the GAN grad beecomes too big, lambda goes down and g_loss weight is reduced. If rec grad is too big, lambda goes down
# and g_loss weight is increased
def calculate_adaptive_weight(vqvae_loss, g_loss, last_layer, epsilon = 1e-6, discriminator_weight = 1.0):
    l_rec_grad = torch.autograd.grad(vqvae_loss, last_layer, retain_graph = True)
    l_gan_grad = torch.autograd.grad(g_loss, last_layer, retain_graph = True)

    adaptive_weight = torch.norm(l_rec_grad) / (torch.norm(l_gan_grad) + epsilon)
    adaptive_weight = torch.clamp(adaptive_weight, 0, 1e4).detach() # clamping prevents explosions and nanloss when the training is initially unstable
    adaptive_weight = discriminator_weight * adaptive_weight
    return adaptive_weight