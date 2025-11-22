# Flow matching

## Process


## 

## summary

1. We can express vector field u_t(x_t) as Expectation of conditional vector field, u_t|1(x_t|x_1).
   
   This process is mathematically proved by comparing the result of fokker-planck eq. of p_t|1 and p_t

2. Predict u_t|1(x_t|x_1) using network instead of u_t(x_t), and use x_{t + delta t} = x_t + (delta t) * network_result to do Inference

<p align="center">
   <img width="662" height="269" alt="스크린샷 2025-11-22 오후 6 54 09" src="https://github.com/user-attachments/assets/19ef97d9-8ce8-487d-857e-6af12c94ece4" />
   <i> https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching </i>
</p>



## Reference
original paper - https://arxiv.org/abs/2210.02747
CS492 - https://mhsung.github.io/kaist-cs492d-fall-2024/
flow matching explanation - https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html
