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


## Process






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



## Reference

original paper  -  https://arxiv.org/abs/1312.6114

Huggingface Diffuser.AutoencoderKL  -  https://huggingface.co/docs/diffusers/api/models/autoencoderkl#diffusers.AutoencoderKL






