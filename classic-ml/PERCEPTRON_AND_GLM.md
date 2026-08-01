### Perceptron

**LOGISTIC REGRESSION VS PERCEPTRON**
#### UPDATING

| Sigmoid/ Logistic Function                     | Step Function                                            |
| ---------------------------------------------- | -------------------------------------------------------- |
| ![[Pasted image 20260801213124.png\|321]]      | ![[Pasted image 20260801213237.png\|306]]<br>            |
| $$g(z) = \frac{1}{1 + e^{-z}}$$                | $$g(z)=\begin{cases}1, & z \ge 0 \\0, & z<0\end{cases}$$ |
| $$h_\theta(x) = \frac{1}{1 + e^{-\theta^Tx}}$$ | $$h_\theta(x) = g[\theta^Tx]$$                           |
#### BATCH GRADIENT ASCENT FOR PERCEPTRON
$$
\theta_j := \theta_j + \alpha\sum_{i=0}^m (y^{(i)}x-h_\theta (x^{(i)}))x_j^{(i)}
$$
when $\alpha\sum_{i=0}^m (y^{(i)}x-h_\theta (x^{(i)}))x_j^{(i)}$ = 0, means the algorithm got the answer right
when $\alpha\sum_{i=0}^m (y^{(i)}x-h_\theta (x^{(i)}))x_j^{(i)}$ = +/- 1, means the algorithm got it wrong
### Exponential Families
Probability Density Function (Continuous Distribution) 
$$
P(y;\eta) = b(y).exp(\eta^T T(y) - a(\eta))
$$
y --> data, $\eta$ --> natural parameter, T(y) --> Sufficient statistic, b(y) --> base measure
a($\eta$) --> log-partition function 
### Examples
#### Bernoulli (Binary Data)
$$
\phi = P(y=1)
$$
$$
P(y;\phi)
=
\phi^y(1-\phi)^{1-y}
$$
$$
=
\exp\left(
\log\left(
\phi^y(1-\phi)^{1-y}
\right)
\right)
$$
$$
=
\exp\left(
\log\left(\frac{\phi}{1-\phi}\right)y
+
\log(1-\phi)
\right)
$$
***Comparing with the exponential family form,***
$$
P(y;\eta)
=
b(y)
\exp\left(
\eta T(y)-a(\eta)
\right),
$$

$$
b(y)=1,
$$

$$
T(y)=y,
$$

$$
\eta
=
\log\left(\frac{\phi}{1-\phi}\right),
$$
$$
\phi
=
\frac{1}{1+e^{-\eta}}.
$$
$$
a(\eta)
=
-\log(1-\phi),
$$
$$
\phi
=
\frac{1}{1+e^{-\eta}},
$$
$$
a(\eta)
=
-\log\left(
1-\frac{1}{1+e^{-\eta}}
\right)
=
\log(1+e^{\eta}).
$$
#### Gaussian (with fixed variance)
