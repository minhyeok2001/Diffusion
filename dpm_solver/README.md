# 🌀 DPM-Solver

DPM-Solver is based on the theoretical result that the reverse SDE of DDPM can be transformed into a deterministic ODE,

which has the same marginal distribution as the original diffusion process. 

This ODE has a first-order semi-linear structure, and by re-parameterizing the time variable into the log-SNR domain, 

its integral term can be computed analytically using Talyor expansion which leads to a high-order solver similar to Runge–Kutta methods. 

Leveraging this property, DPM-Solver plugs in the noise predictor ε which was trained in DDPM (w/ cfg),  

and constructs first-, second-, and third-order high-accuracy ODE solvers, enabling the model to estimate x(t) with very few sampling steps.


## Process

![DPM solver-2](https://github.com/user-attachments/assets/896f85e5-e520-4fd3-9e2c-9dc747d84b39)
![DPM solver-3](https://github.com/user-attachments/assets/efe7a650-db6b-42b2-a4fa-8ba016950a5e)
![DPM solver-4](https://github.com/user-attachments/assets/45cd3dc7-db90-41be-8a6e-8305a6fc5dad)
![DPM solver-5](https://github.com/user-attachments/assets/ac7ff5f7-d7b3-42b0-aee5-141f59d4ffcf)


<p align="center">
  <img width="543" height="591" alt="스크린샷 2025-11-13 오후 5 54 37" src="https://github.com/user-attachments/assets/01cd57dd-b619-4bcc-8a09-a63945b2330d" /> <br>
  <i>High-order ODE solver algorithm from DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps</i>
</p>


## Issues encountered





## Reference
- original paper - https://arxiv.org/abs/2206.00927
- CS492 - https://mhsung.github.io/kaist-cs492d-fall-2024/


