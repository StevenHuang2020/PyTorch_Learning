# PyTorch_Learning
![License: MIT](https://img.shields.io/badge/License-MIT-blue)
![Python Version](https://img.shields.io/badge/Python-v3.6-blue)
![PyTorch Version](https://img.shields.io/badge/PyTorch-V1.7-brightgreen)

PyTorch learning.
 - PyTorch: https://pytorch.org/
 - DOC: https://pytorch.org/docs/stable/index.html
 - Neural Layers: https://pytorch.org/docs/stable/nn.html
 
## Regression Result
|||
|---|---|
|<img src="images/curve0.gif" width="320" height="240" />|<img src="images/curve1.gif" width="320" height="240" />|

## MNIST Classification Result
|||
|---|---|
|<img src="images/loss.png" width="320" height="240" />|<img src="images/cm.png" width="320" height="240" />|

## Optimization algorithms
<img src="images/lossDict.png" width="320" height="240" />

## PyTorch Loss Functions 
<!-- ![equation](https://latex.codecogs.com/svg.latex?\Large&space;x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}) -->

The mean squared error loss(MSELoss):<br/>
<img src="https://latex.codecogs.com/svg.latex?\begin{equation}%20\begin{array}{l}%20\ell(x,y)=L=\left\{l_1,...,l_N%20%20\right\}^\top,%20l_n=(x_n-y_n)^2\\%20\ell(x,y)=%20\begin{cases}%20mean(L),&%20\text{if%20reduction=%27mean%27;}\\%20sum(L),&%20\text{if%20reduction=%27sum%27;}%20\end{cases}%20\end{array}%20\end{equation}"/>

The Binary Cross Entropy(BCELoss):<br/>
<img src="https://latex.codecogs.com/svg.latex?\begin{equation}%20\begin{array}{l}%20\ell(x,y)=L=\left\{l_1,...,l_N%20%20\right\}^\top,%20l_n=-w_n[y_n\cdot%20\log{x_n}+(1-y_n)\cdot%20\log{1-x_n}]\\%20\ell(x,y)=%20\begin{cases}%20mean(L),&%20\text{if%20reduction=%27mean%27;}\\%20sum(L),&%20\text{if%20reduction=%27sum%27;}%20\end{cases}%20\end{array}%20\end{equation}"/>
 
The Kullback-Leibler divergence loss(KLDivLoss):<br/>
<img src="https://latex.codecogs.com/svg.latex?\begin{equation}%20\begin{array}{l}%20\ell(x,y)=L=\left\{l_1,...,l_N%20%20\right\}^\top,%20l_n=y_n\cdot(\log{y_n}-x_n)\\%20\ell(x,y)=%20\begin{cases}%20mean(L),&%20\text{if%20reduction=%27mean%27;}\\%20sum(L),&%20\text{if%20reduction=%27sum%27;}%20\end{cases}%20\end{array}%20\end{equation}"/>

## Output size calculation
 - Convolutional Layer <br/> 
 <img src="https://latex.codecogs.com/svg.latex?O=\frac{W%20-%20F%20+%202P}{S}+1" /> ,where W=Input image size, F=kernel size, P=padding, S=stride
 - Pooling Layer  <br/>
 Same as the Convolutional Layer
 
## Parameter numbers calculation
 - Input layer  <br/>
  <img src="https://latex.codecogs.com/svg.latex?parameters%20=%200" /> <br/>
 - Fully Connected Layer (FC)  <br/>
  <img src="https://latex.codecogs.com/svg.latex?((c%20*%20p)%20+%201*c)" /> <br/>
  <img src="https://latex.codecogs.com/svg.latex?c*p" />: weights, <img src="https://latex.codecogs.com/svg.latex?1*c" />: bias iterm.
  c: current layer neurons, p: previous layer neurons.<br/>
 - Convolutional Layer  <br/>
 <img src="https://latex.codecogs.com/svg.latex?((m%20*%20n%20*%20d)+1)*%20k)" /> <br/>
 m: width of the filter, n: height of the filter, d: input filter numbers, k: ouptut filter numbers. <br/>
 - Pooling Layer  <br/>
 <img src="https://latex.codecogs.com/svg.latex?parameters%20=%200" /> <br/>