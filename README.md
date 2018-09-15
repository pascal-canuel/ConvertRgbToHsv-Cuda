<h1 align="center">Convert RGB2HSV - Cuda</h1> 
<p align="center">
<img src="https://img.shields.io/badge/License-MIT-blue.svg">
</p>

<p align="center">🌈 Parallelization and implementation of the algorithm to convert RGB pictures to HSV in CUDA. 🌈 </p>

<p align="center">Algorithm</p>
1. The R,G,B values are divided by 255 to change the range from 0..255 to 0..1:
R' = R/255   
G' = G/255   
B' = B/255   
2. 
Cmax = max(R', G', B')
Cmin = min(R', G', B')

Δ = Cmax - Cmin
 
Hue calculation:

Saturation calculation:

Value calculation:

V = Cmax
## By Pascal Canuel
