 We want $h_\theta$ (x) $\epsilon$ [0, 1], That means the prediction of the model must be one of these.
# $h_\theta$ (x) = g($\theta^T$x) = $\frac{1}{1 + e^{-\theta^Tx}}$ 

### Sigmoid Function
#### g(z) = $\frac{1}{1 + e^{-z}}$

### Probability of classifications -
$P$ (y = 1 | x ; $\theta$) = $h_\theta$ (x) 
$P$ (y = 0 | x ; $\theta$) = 1 - $h_\theta$ (x) 
y $\epsilon$  {0, 1}

#### $P$ (y | x ; $\theta$) = $h_\theta$(x)$^y$.(1 - $h_\theta$ (x))$^{1-y}$ 
### Maximum Likelihood Estimation
$$
L(\theta) = P(\vec{y}\mid X;\theta)
= \prod_{i=1}^{m} P\left(y^{(i)} \mid x^{(i)};\theta\right)
= \prod_{i=1}^{m} \left(h_{\theta}(x^{(i)})\right)^{y^{(i)}}
\left(1 - h_{\theta}(x^{(i)})\right)^{1-y^{(i)}}
$$
### Log Likelihood
$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^{m} \left[ y^{(i)} \log h_\theta(x^{(i)}) + (1-y^{(i)}) \log(1-h_\theta(x^{(i)})) \right]$$
we have to choose parameters $\theta$ to maximize $l(\theta)$.
### Algorithm used to maximize
***Batch Gradient Ascent***
$$
\theta_j := \theta_j + \alpha\frac{\partial}{\partial\theta_j}l(\theta)
$$
$$
\theta_j := \theta_j + \alpha\sum_{i=0}^m (y^{(i)}-h_\theta (x^{(i)}))x_j^{(i)}
$$
### Newton's Method
$$
\theta^{(t+1)} = \theta^{(t)} - \frac{f'\left(\theta^{(t)}\right)}{f''\left(\theta^{(t)}\right)}
$$
$$
\theta^{(t+1)} = \theta^{(t)} - H^{-1}\nabla J\left(\theta^{(t)}\right)
$$
$$
\theta := \theta - H^{-1}\nabla J(\theta)
$$
$$
H_{ij} = \frac{\partial^2 J(\theta)}{\partial \theta_i \partial \theta_j}
$$




