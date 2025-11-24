# Flow matching

## Process
![Flow matching-1](https://github.com/user-attachments/assets/ebc549ba-2639-4132-96f2-4f7068811434)
![Flow matching-2](https://github.com/user-attachments/assets/b307379d-a09c-4b7f-8d0b-96f73f4e1b3a)
![Flow matching-3](https://github.com/user-attachments/assets/3f3f193a-28b5-43eb-8aa2-8fa0375e9536)
![Flow matching-4](https://github.com/user-attachments/assets/f81ed349-f055-4d16-af19-ebfba70455b5)
<img width="1536" height="1000" alt="image" src="https://github.com/user-attachments/assets/2d757bc7-20de-4948-9b40-2cfb2960de5c" />

## summary

1. We can express vector field u_t(x_t) as Expectation of conditional vector field, u_t|1(x_t|x_1).
   
   This process is mathematically proved by comparing the result of fokker-planck eq. of p_t|1 and p_t

2. Predict u_t|1(x_t|x_1) using network instead of u_t(x_t), and use x_{t + delta t} = x_t + (delta t) * network_result to do Inference

<p align="center">
   <img width="662" height="269" alt="스크린샷 2025-11-22 오후 6 54 09" src="https://github.com/user-attachments/assets/19ef97d9-8ce8-487d-857e-6af12c94ece4" /><br>
   <i> https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching </i>
</p>

## Experimental result

13 hours training in Google Colab, using a single A100.
<p align="center">
<img width="750" height="304" alt="스크린샷 2025-11-24 오후 8 31 11" src="https://github.com/user-attachments/assets/686c867a-057a-44f7-90b2-1dc51540b9ee" /><br>
   <i> loss curve - w&b </i>
</p>


<p align="center">
<img width="640" height="128" alt="image" src="https://github.com/user-attachments/assets/f2bbacf9-cedc-4a03-ab3f-a86e673514af" /><br>
<img width="640" height="128" alt="image" src="https://github.com/user-attachments/assets/7dfeac0c-4227-4a79-a101-848db25d6564" /><br>
   <i> Images from 20 step sampling (4steps interval)</i>
</p>



**Table 1. Result of sampling using Optimal transport**

<table>
  <tr>
    <th> sampling steps </th>
    <th>FID</th>
  </tr>
  <tr>
    <td>5</td>
    <td>294.72</td>
  </tr>
  <tr>
    <td>10</td>
    <td><b>206.12</b></td>
  </tr>
  <tr>
    <td>20</td>
    <td>249.16</td>
  </tr>
  <tr>
    <td>50</td>
    <td>244.37</td>
  </tr>
  <tr>
    <td>100</td>
    <td>243.42</td>
  </tr>
</table>



## Reference
original paper - https://arxiv.org/abs/2210.02747

CS492 - https://mhsung.github.io/kaist-cs492d-fall-2024/

flow matching explanation - https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html
