# ⚙️ Denoising Diffusion Models with Classifier Free Guidance

## Process

## Loss Function & Reverse Process & Sampling Process Derivation

### 1. DDPM
<img width="1000" height="1478" alt="image" src="https://github.com/user-attachments/assets/4694507a-be63-4d96-944d-ec44fd7bfb00" />
<img width="1000" height="1175" alt="image" src="https://github.com/user-attachments/assets/75dac97e-61f2-40a3-891f-2e1130b2a588" />

### 2. DDIM
<img width="1000" height="1216" alt="image" src="https://github.com/user-attachments/assets/8281a16b-77df-4d19-81e1-fe802a6df6e8" />


### 3. CFG
<img width="1000" height="1062" alt="image" src="https://github.com/user-attachments/assets/5bcadef3-4801-4ca0-8f06-faec95d26296" />



## Issues encountered 

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




