# 🎨 Stable Diffusion 
<p align="center">
  <img width="822" height="420" alt="스크린샷 2025-12-02 오후 3 37 29" src="https://github.com/user-attachments/assets/1a2cb235-36d3-4b0c-9474-2bf61c4ffa7a" />
</p>

Stable Diffusion is a diffusion model that performs the denoising process in a latent space learned by an autoencoder, rather than directly in pixel space.

The authors argue that operating in pixel space is computationally inefficient, 

because a large portion of pixel-level information consists of perceptually irrelevant details. 

Diffusion models must still process all pixels during both training and sampling, which leads to unnecessary computation.

To address this, they introduce a perceptual autoencoder that compresses the image into a latent space that preserves the semantic and perceptually important structure, 

while discarding high-frequency details that humans cannot perceive well.

In this section, I focus on the aspects that are unique to Stable Diffusion, since VAE-style autoencoders, DDPM, and CFG are already covered in other parts of this repository.

There are a few key points worth noting in Stable Diffusion.

### 1. VAE objectives

<p align="center">
  <img width="1005" height="88" alt="스크린샷 2025-12-02 오후 3 50 24" src="https://github.com/user-attachments/assets/782b6223-bd73-454d-9c26-0af535f4ddc9" />
  <i>Loss function of AutoEncoder in G. Details on Autoencoder Models</i>
</p>

The full objective consists of four main components:

----

**Reconstruction term**

The first part is a combination of a pixel-based reconstruction loss and the LPIPS perceptual loss.

LPIPS (Learned Perceptual Image Patch Similarity), proposed in “The Unreasonable Effectiveness of Deep Features as a Perceptual Metric”, 

measures similarity in a deep feature space (e.g., using VGG), rather than directly in pixel space.

This is motivated by the fact that L1 and L2 losses do not match human perceptual judgment and often result in overly smooth or blurry reconstructions.

LPIPS encourages the autoencoder to preserve semantic and perceptually meaningful structures.

----

**Adversarial loss**

The second and third terms correspond to the adversarial objective, similar to the one used in GANs.

The purpose of the GAN loss here is to ensure that the reconstructed images lie on the manifold of real images by enforcing local realism through a patch-based discriminator.

This helps prevent blurriness, a well-known issue when only reconstruction losses are used (as in standard VAEs).

This combination produces reconstructions that are sharper and more realistic than those produced by VAEs trained only with pixel losses.

----

**KL regularization term**

The last term is a lightly weighted KL divergence, regulating the latent distribution q to stay close to a standard normal distribution.

Unlike classical VAEs, this KL term is used with very small weight in practical aspects.

### 2. How to inject conditions ?


### 3. 

## Reference
original paper - https://arxiv.org/abs/2112.10752
LPIPS - https://arxiv.org/abs/1801.03924
