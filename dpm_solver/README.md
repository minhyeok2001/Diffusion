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

### 1. Changing code implementation style while finding t_prev

In DDIM, when inference timestep was given as 100 when total num_trainstep is 1000, I used to implement with the code below.
```
t_prev = (t-1)*self.ratio
t = t * self.ratio
```
but with this method, I found that it does not exactly start from 999 time.

So instead, I used torch.linspace() for DPM-solver

### 2. Finding corresponding midpoint time in 2nd-order DPM-solver

In original paper, It suggests to use inverse mapping using interpolation to solve midpoint time.

Instead, In my code, I simply select the closest discrete timestep

```
distance = torch.abs(self.dpm_lambdas - lambda_mid)
mid = torch.argmin(distance)
```
This is a practical discrete approximation of the continuous interpolation.

## Experimental result

Remember DDPM w/ cfg? It takes about 1 hour to eval with 1000 steps sampling. 

But with DPM-Solver, It takes only 20 seconds with 5 steps sampling, even with better performance !! 

<p align="center">
  <img width="522" height="132" alt="image" src="https://github.com/user-attachments/assets/ab610cfe-6e13-46ed-9dd5-4663dec96d1e" /> <br>
  <i>5step results on 1st-order approximation</i>
</p>


<p align="center">
  <img width="522" height="132" alt="image" src="https://github.com/user-attachments/assets/66855b73-5229-472a-8649-d0ca800ec0f2" /> <br>
  <i>10step results on 1st-order approximation</i>
</p>

--- 

**Table 1. result of 1st-order & 2nd-order approximation**

<table>
  <tr>
    <th> sampling steps </th>
    <th>1st order</th>
    <th>2nd order</th>
  </tr>
  <tr>
    <td>5</td>
    <td><b>13.62</b></td>
    <td>45.17</td>
  </tr>
  <tr>
    <td>10</td>
    <td>36.03</td>
    <td>103.21</td>
  </tr>
  <tr>
    <td>20</td>
    <td>60.52</td>
    <td>77.78</td>
  </tr>
  <tr>
    <td>50</td>
    <td>71.79</td>
    <td>93.11</td>
  </tr>
  <tr>
    <td>100</td>
    <td>86.52</td>
    <td>68.12</td>
  </tr>
</table>

In the N-order case, the NFE is 2 * N * inference_step ( multiplying 2 is from CFG )

## Points to consider

**1. Why does the FID score of my DPM-solver get worse when the NFE becomes larger?**

As you can see the result on Table 1 above, DPM-solver get worse when NFE becomes larger in 1st-order approximation. 

Cannot find any reason for this,... Maybe it’s because my dataset is quite small, which could introduce bias. FID tends to work best on large datasets...


## Reference
- original paper - https://arxiv.org/abs/2206.00927
- CS492 - https://mhsung.github.io/kaist-cs492d-fall-2024/





