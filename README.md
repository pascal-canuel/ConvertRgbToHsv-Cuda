<h1 align="center">Convert RGB2HSV - Cuda</h1> 
<p align="center">
<img src="https://img.shields.io/badge/License-MIT-blue.svg">
</p>

<p align="center">🌈 Parallelization and implementation of the algorithm to convert RGB pictures to HSV in CUDA. 🌈 </p>

<h1 align="center">Algorithm</h1> 
1. The R,G,B values are divided by 255 to change the range from 0..255 to 0..1:<br>
R' = R/255<br>
G' = G/255<br> 
B' = B/255<br><br>  
2. Find the maximum & minimum<br>
Cmax = max(R', G', B')<br>
Cmin = min(R', G', B')<br><br>
3. Find delta<br>
Δ = Cmax - Cmin<br><br>
4. Hue calculation:
<img src="https://www.rapidtables.com/convert/color/rgb-to-hsv/hue-calc2.gif"/><br>
5. Saturation calculation:
<img src="https://www.rapidtables.com/convert/color/rgb-to-hsv/sat-calc.gif"/><br>
6. Value calculation:<br>
V = Cmax

## By Pascal Canuel
