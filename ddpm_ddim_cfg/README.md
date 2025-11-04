# ⚙️ Denoising Diffusion Models with Classifier Free Guidance
<img width="1200" height="400" alt="스크린샷 2025-11-04 오후 3 51 38" src="https://github.com/user-attachments/assets/fd2f2368-af8c-4e9b-ad80-68d423016533" />

## Denoising Diffusion Probabilistic Model

The goal of diffusion models is to approximate the real data distribution p(x) by learning to reverse a noise process.

Conceptually, it’s similar to a Variational Autoencoder, but diffusion models leverage the Markov property over discrete timesteps.

A single U-Net is used as the denoising network, and its output is the predicted noise ε.

There are two main parts in DDPM procedure, one is **forward process** and the other is **reverse process**.

### Forward process

The forward process gradually adds Gaussian noise to clean data x_0 using a predefined noise schedule **β**. 

Thanks to markov property, we can move from any timestep T to 0 in one step using a closed-form equation.

### Reverse process

Reverse process is just a inversion of forward process, and we can derive its formula from bayes' rule.

With the predicted noise from Unet, we can approximate the distribution of P(x_t-1 | x_t) via reparameterization trick.

By iteratively applying this step from t = T to 0, we can sample clean image from p(x_0 | x_1) 

### Training method

Training is straightforward:

	1.	Randomly sample timesteps t for each batch.
  
	2.	Add the corresponding level of noise to the input.
  
	3.	Train the U-Net to predict the added noise at each timestep.

### Samlpling method

Sampling is straightforward, either :

  1. Start from pure noise at timestep t=T, predict noise using Unet and find x_0 using forward formula, predicted noise.

  2. Sample x_t-1 from p(x_t-1 | x_t) using x_0 from above using reparameterization trick

  3. iter 1-2 process from t=T to t=0

## Denoising Diffusion Implicit Model

DDIM  is an advanced variant of DDPM that enables deterministic sampling,

requiring only a few denoising steps to generate high-quality samples.

DDIM has the same forward process and marginal probability with DDPM, but the Reverse process is slightly different from DDPM.

unlike derive P(x_t-1 | x_t) using bayes' rule from forward process , 

DDIM 's first assumption is that maybe we can express the reverse step using the linear combination of x_0 and x_T, without any kind of markov property.

Unlike DDPM, which defines P(x_t-1 | x_t) using Bayes’ rule under the Markov assumption,

DDIM starts from a different assumption — that the reverse transition can be expressed as a linear combination of x_0 and x_T without relying on the Markov property, 

which leads to a surprisingly short inference time.

We can re-use the Unet model that we trained for DDPM, _the only difference is the sampling step_

## Classifier Free Guidance

This method originates from Classifier Guidance, 

which combines a noise-prediction model with an external classifier that takes x_t as input to guide the generation toward a specific class.

Through mathematical analysis, researchers found that **Classifier Free Guidance** is equivalent in effect to Classifier Guidance, 

which does not use any additional classifier.

The detailed mathematical derivations are provided above.

## Loss Function & Reverse Process & Sampling Process Derivation

### 1. DDPM
<img width="1000" height="1478" alt="image" src="https://github.com/user-attachments/assets/4694507a-be63-4d96-944d-ec44fd7bfb00" />
<img width="1000" height="1175" alt="image" src="https://github.com/user-attachments/assets/75dac97e-61f2-40a3-891f-2e1130b2a588" />

### 2. DDIM
<img width="1000" height="1216" alt="image" src="https://github.com/user-attachments/assets/8281a16b-77df-4d19-81e1-fe802a6df6e8" />


### 3. CFG
<img width="1000" height="1062" alt="image" src="https://github.com/user-attachments/assets/5bcadef3-4801-4ca0-8f06-faec95d26296" />



## Issues encountered 

### DDPM

**1. Loss spike**

<img width="700" height="500" alt="스크린샷 2025-11-04 오후 3 17 09" src="https://github.com/user-attachments/assets/b5d6557a-4168-42f3-9e61-710a6390873c" />

A noticeable loss spike was observed when using a learning rate of 5e-4.

After several trials, I found that 5e-5 works better for my task, as shown above.

**2. Poor reverse diffusion quality**

<p align="center">
  <img width="900" height="280" alt="asdasasd" src="https://github.com/user-attachments/assets/1710d7ff-f102-490a-ae8c-d6dc3a666541" /><br>
  <i>Result of 500 steps of learning with LR 5e-5. 1000 -> 900 -> ... -> 100 -> 0 timesteps result from left side</i>
</p>


Although both training and validation losses converged properly, the reverse diffusion quality was noticeably poor.

To identify the root cause, I inspected multiple factors — including the timesteps, alpha schedule, and both forward and reverse formulas.

<p align="center">
  <img width="900" height="280" alt="image" src="https://github.com/user-attachments/assets/82be0f4f-3791-4cbd-b61c-df25714379ca" /><br>
  <i>Check forward step</i>
</p>


<p align="center">
  <img width="900" height="280" alt="hahahahahaha" src="https://github.com/user-attachments/assets/60c9bb60-5c93-40c5-ae78-168496553276" /><br>
  <i>Check reverse step using "faverogian/Smithsonian128UNet - huggingface" as a eps predicting network. <br> Other factors are all same with my original one</i>
</p>

Based on this comparison, I found that when replacing my U-Net with the open-source model, the results improved significantly — implying that the issue likely stems from either ...
**(1) insufficient training time** or **(2) a poorly designed model architecture.**

So far, I’ve trained my model with 15,000 128×128 images for about 12 hours on an A100, which seems like enough training time based on my experience in CS492.

So I double checked my Unet model







## Reference
- original paper - https://arxiv.org/abs/2006.11239
- Mathematical approach - https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
- CS492 - https://mhsung.github.io/kaist-cs492d-fall-2024/






