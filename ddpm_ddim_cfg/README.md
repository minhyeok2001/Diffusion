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

### 1. Loss spike

<p align="center">
	<img width="700" height="500" alt="스크린샷 2025-11-04 오후 3 17 09" src="https://github.com/user-attachments/assets/b5d6557a-4168-42f3-9e61-710a6390873c" />
	<i>Training loss on WnB</i>
</p>

A noticeable loss spike was observed when using a learning rate of 5e-4.

After several trials, I found that 5e-5 works better for my task, as shown above.

---
### 2. Poor reverse diffusion quality

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

I modified my model architecture focusing on three aspects -

a symmetric encoder–decoder design, skip connections between the encoder and decoder, and replacing the original single-head attention with multi-head attention. 

The original version was largely based on a VAE-UNet architecture, which is asymmetric — the decoder is thicker than the encoder — and lacks skip connections between them.

The results below are from 200 epochs of training, taking about 14 hours on an A100. 

For more details, check [Run named "DDPM" on W&B](https://wandb.ai/mhroh01-ajou-university/Diffusion/table?nw=nwusermhroh01).

<p align="center">
  <img width="700" height="500" alt="스크린샷 2025-11-05 오후 3 34 08" src="https://github.com/user-attachments/assets/4786ff41-a470-49e7-a10e-0250a0b4acc5" /><br>
  <i>Training & validation loss on WnB</i>
</p>

<p align="center">
  <img width="1280" height="128" alt="image" src="https://github.com/user-attachments/assets/8fc379de-a4f7-48b7-ab77-74827ad19d05" /><br>
	<img width="1280" height="128" alt="image" src="https://github.com/user-attachments/assets/46f56ae4-75d6-4f61-919d-5c0dadacb455" /><br>
	<img width="1280" height="128" alt="image" src="https://github.com/user-attachments/assets/a5b0111b-603a-4cf5-9afa-7af9b6747217" /><br>
  <i>Inference from random noise</i>
</p>

---
### 3. DDIM code implementation issues (Mathematical derivation, timestep handling)

When using subsampled timesteps in DDIM — for example, running 100 steps instead of the full 1000 —
the _alpha_ values should be **derived from cumprod_alphas**, rather than directly using the original alpha values from DDPM.

This is typically implemented in code as follows:
```
t_prev = (t-1)*self.ratio
t = t * self.ratio
t_prev_safe = t_prev.clamp(min=0)
a_prev_vals = self.cumprod_alpha[t_prev_safe]
```

---
### 4. Mean predictor training issues

Even though the DDPM framework allows training a single network to predict noise, x_0, or the mean, our experiments show that the mean predictor is highly unstable.

Both the **noise predictor and the x_0 predictor converge normally**, but the **mean predictor consistently fails to learn and collapses** to producing almost black samples.

I found that it aligns to the observations from the original DDPM paper: 

the authors reported that training a mean predictor with a simple MSE loss does not converge,

indicating that mean prediction requires additional constraints or different loss formulations to work properly.

<p align="center">
  <img width="899" height="362" alt="스크린샷 2025-11-19 오전 11 29 04" src="https://github.com/user-attachments/assets/37174adc-c26f-4e47-96c4-12a001c2315b" /><br>
<i>Mean predictor network does converge</i>
</p>


<p align="center">
  <img width="350" height="350" alt="스크린샷 2025-11-19 오전 10 56 48" src="https://github.com/user-attachments/assets/29607972-fa32-421b-80bd-3da4c5c71a10" /><br>
<i>The comparison of mean predictor on the original DDPM paper</i>
</p>

The comparison table is on the below, Table 3.

---
## Experimental result 

**Table 1. FID Scores of DDPM Models (with / without Classifier-Free Guidance)**

<table>
  <tr>
    <th>Model</th>
    <th>FID Score</th>
  </tr>
  <tr>
    <td>DDPM w/o cfg</td>
    <td>172.30</td>
  </tr>
  <tr>
    <td>DDPM w/ cfg (cfg_weight = 1.5)</td>
    <td>48.85</td>
  </tr>
  <tr>
    <td>DDPM w/ cfg (cfg_weight = 2.0)</td>
    <td>55.04</td>
  </tr>
  <tr>
    <td>DDPM w/ cfg (cfg_weight = 2.5)</td>
    <td><b>23.99</b></td>
  </tr>
  <tr>
    <td>DDPM w/ cfg (cfg_weight = 3.0)</td>
    <td>37.45</td>
  </tr>
</table>

----

**Table 2. FID Scores of DDIM Models by Inference Step and η Value**


<table>
  <tr>
    <th>Inference Step</th>
    <th>DDIM (η=1.0)</th>
    <th>DDIM (η=0.5)</th>
    <th>DDIM (η=0.0)</th>
  </tr>
  <tr>
    <td>100</td>
    <td>369.03</td>
    <td>476.34</td>
    <td>589.71</td>
  </tr>
  <tr>
    <td>250</td>
    <td>403.41</td>
    <td>609.96</td>
    <td>778.55</td>
  </tr>
  <tr>
    <td>500</td>
    <td>394.72</td>
    <td>473.83</td>
    <td>593.69</td>
  </tr>
  <tr>
    <td>750</td>
    <td>275.70</td>
    <td>200.49</td>
    <td>302.95</td>
  </tr>
  <tr>
    <td>1000</td>
    <td><b>155.86</b></td>
    <td>159.62</td>
    <td>251.57</td>
  </tr>
</table>

----

**Table 3. Comparison of noise / mean / x_0 predictor**



<table>
  <tr>
    <th>noise predictor</th>
    <th>mean predictor</th>
    <th>x_0 predictor</th>
  </tr>
  <tr>
    <td><b>172.30</b></td>
    <td>401.41 (loss does not converge, produces black images)</td>
    <td>258.32</td>
  </tr>
</table>

----
**Fig 1. Results of DDIM with each step, eta**
<p align="center">
  <img width="968" height="1190" alt="image" src="https://github.com/user-attachments/assets/7540a5bb-0648-4cc1-904e-8204cbcf3f2e" /><br>
</p>

## Points to consider 

1.	Why do my DDIM results underperform, even though the paper suggests ~100 sampling steps are sufficient? It seems like 750 steps images are fine, but ~500 steps images are not good
<p align="center">
	<img width="1133" height="396" alt="스크린샷 2025-11-11 오후 4 11 14" src="https://github.com/user-attachments/assets/31d6ddca-43cd-4121-8b0c-9f5c4ebc7b10" />
	<i> Experiments from denoising diffusion implicit model, ICLR 2021</i>
</p>

2.	Why is there such a large performance gap between runs without CFG and with CFG?


3.	According to Fig 1, with respect to the sampling timestep size, denoising is described as progressing from coarse structure to fine details. How can we explain this?
	   
	In my opinion, points 1 and 2 depend heavily on the dataset and the number of validation images, which seems reasonable to me.
	
	But point 3, I guess that when the timestep interval is too large, each denoising step has to reconstruct too much information at once, making the process overly coarse.

4. Why mean predictor does not work(converge)?



## Reference
- original paper - https://arxiv.org/abs/2006.11239
- Mathematical approach - https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
- CS492 - https://mhsung.github.io/kaist-cs492d-fall-2024/



















