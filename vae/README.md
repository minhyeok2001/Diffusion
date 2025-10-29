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

## Issues encountered during training

1. LR은 0.001, 0.0001보다  0.00005 제일 좋음을 확인

<img width="1030" height="1030" alt="unknown" src="https://github.com/user-attachments/assets/f5b7f1f5-8d2f-472d-935a-9803ba635a52" />

2. 형체는 비슷하지만 여전히 FID가 낮아서, loss의 kl divergence term과 reconstruction term scaling 시도 -> matching term에 0.3 곱하여 더하기 -> 에폭을 늘릴수록 이미지가 흐려짐을 발견


<img width="1030" height="1030" alt="image" src="https://github.com/user-attachments/assets/e073e69f-8d17-4a47-b2f8-e04549bde3db" />
< 12시간 학습시 결과 >  , Fid score = 600


<img width="522" height="262" alt="image" src="https://github.com/user-attachments/assets/6318409b-95a8-4fce-9e62-b2bf39019ccd" />

6hour on a100  Fid score = 149.12
## Reference

original paper  -  https://arxiv.org/abs/1312.6114

Huggingface Diffuser.AutoencoderKL  -  https://huggingface.co/docs/diffusers/api/models/autoencoderkl#diffusers.AutoencoderKL

Q1's reference - https://www.datacamp.com/tutorial/variational-autoencoders










