# 🧩 Variantional Auto Encoder

<img width="1700" height="700" alt="image" src="https://github.com/user-attachments/assets/dc3277c6-fb40-4859-9207-3323bb4b207f" />

Our goal is to approximate the real data distribution p(x) using a neural network.

Neural network consists of two different network, Encoder and Decoder.

Encoder predicts mean and standard deviation of latent field, while Decoder only predicts mean with its standard deviation fixed to 1.

In detail, we predict mean and std of each component in latent space. 

8x64x64 encoder result = cat [ mu 4x64x64 , sigma 4x64x64 ]

x stands for real dataset. What we should do is to maximize the probability of data distribution p(x). 

That is, Probability p(x) should be high for real data input x, which means it is plausible. 

**-> -log p(x) can play a role as a loss function ..**

## Structure

Since the original Variational Encoder paper doesn’t provide an official GitHub implementation, 

I decided to adopt the architecture used in Hugging Face’s diffusers.AutoencoderKL.


## Loss Function Derivation

$$
\begin{align*}
\mathcal{Loss} 
&= - \mathbb{E}_{q_\phi(z|x)} [ \log p_\theta(x|z) ] 
    + D_{KL}\big(q_\phi(z|x) \||\ p(z)\big)
\end{align*}
$$


<img width="1613" height="1000" alt="image" src="https://github.com/user-attachments/assets/0779af0f-3de2-4c1a-8013-1369b1440ce3" />

<img width="999" height="1357" alt="image" src="https://github.com/user-attachments/assets/f7e4f469-cd13-40b7-b061-ae9091c0e8fa" />


 ### -> Q. Why is reconstruction term intractable while matching term isn't ?
![IMG_7CD0FA3F711D-1](https://github.com/user-attachments/assets/13b57411-e09d-406d-b3cb-6d2914df9b4b)

## Experiments & Ablation Study

- 6 hours in Colab using A100
```bash
 python -m vae.train --batch_size 32 --lr 0.0001 --epoch 150 
```

Due to the relatively low FID score, I referred to the paper “β-VAE”.

The paper suggests that introducing a β coefficient on kl divergence term in loss function helps the model learn a more disentangled representation, leading to better performance.

Therefore, I conducted an ablation study comparing models with and without the β term, and the results are shown above.

All training process, results are available in **_VAE.ipynb_** and **_VAE_No_beta_ver.ipynb_**


<table>
  <tr>
    <th>Model</th>
    <th>FID Score</th>
    <th>Visualization ( randomly sampled in train set )</th>
  </tr>
  <tr>
    <td>w/o β coefficient</td>
    <td>388.65</td>
    <td>
        <div align="center">    
          <img width="250" alt="w/o beta" src="https://github.com/user-attachments/assets/4b4555cd-1cfa-42cb-9eed-3b202a7db5fb" />
    </td>
  </tr>
  <tr>
    <td>w/ β coefficient (β=0.3)</td>
    <td>149.12</td>
    <td>
        <div align="center">
          <img width="250" alt="with beta" src="https://github.com/user-attachments/assets/6318409b-95a8-4fce-9e62-b2bf39019ccd" />
    </td>
  </tr>
</table>

            
## Reference

original paper  -  https://arxiv.org/abs/1312.6114

Huggingface Diffuser.AutoencoderKL  -  https://huggingface.co/docs/diffusers/api/models/autoencoderkl#diffusers.AutoencoderKL

Q's reference - https://www.datacamp.com/tutorial/variational-autoencoders

β vae paper - https://openreview.net/forum?id=Sy2fzU9gl











