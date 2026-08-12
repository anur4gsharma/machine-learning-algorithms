# Perceptron and Generalized Linear Models

## Perceptron

### Logistic Regression vs. Perceptron

| Logistic / Sigmoid Function               | Perceptron / Step Function                               |
| ----------------------------------------- | -------------------------------------------------------- |
| $g(z)=\frac{1}{1+e^{-z}}$                 | $g(z)=\begin{cases}1, & z \ge 0 \ 0, & z < 0\end{cases}$ |
| $h_\theta(x)=\frac{1}{1+e^{-\theta^T x}}$ | $h_\theta(x)=g(\theta^T x)$                              |

The main difference is that logistic regression produces a probability in the range $[0,1]$, while the perceptron produces a binary prediction.

### Batch Gradient Ascent for the Perceptron

The parameter update is

$$
\theta_j := \theta_j + \alpha \sum_{i=0}^{m}
\left(y^{(i)}-h_\theta(x^{(i)})\right)x_j^{(i)}
$$

where:

* $\alpha$ is the learning rate.
* $y^{(i)}$ is the true label.
* $h_\theta(x^{(i)})$ is the predicted label.
* $x_j^{(i)}$ is the $j$-th feature of example $i$.

For the step-function classifier,

$$
y^{(i)}-h_\theta(x^{(i)}) \in {-1,0,1}.
$$

* If the prediction is correct, the error is $0$ and there is no update for that example.
* If the prediction is incorrect, the error is either $+1$ or $-1$, depending on the true label and prediction.

---

## Exponential Families

A probability distribution belongs to the exponential family if it can be written in the form

$$
P(y;\eta)
=========

b(y)\exp\left(\eta^T T(y)-a(\eta)\right)
$$

where:

* $y$ = observed data
* $\eta$ = natural parameter
* $T(y)$ = sufficient statistic
* $b(y)$ = base measure
* $a(\eta)$ = log-partition function

### Bernoulli Distribution

For binary data,

$$
y\in{0,1}
$$

Let

$$
\phi=P(y=1).
$$

The Bernoulli probability mass function is

$$
P(y;\phi)
=========

\phi^y(1-\phi)^{1-y}.
$$

Rewrite this using an exponential:

$$
P(y;\phi)
=========

\exp\left(
\log\left(\phi^y(1-\phi)^{1-y}\right)
\right).
$$

Using logarithm rules,

$$
P(y;\phi)
=========

\exp\left(
y\log\left(\frac{\phi}{1-\phi}\right)
+
\log(1-\phi)
\right).
$$

Comparing this with the exponential-family form

$$
P(y;\eta)
=========

b(y)\exp\left(\eta T(y)-a(\eta)\right),
$$

we obtain

$$
b(y)=1,
$$

$$
T(y)=y,
$$

and

$$
\eta
====

\log\left(\frac{\phi}{1-\phi}\right).
$$

Solving for $\phi$,

$$
\phi
====

\frac{1}{1+e^{-\eta}}.
$$

Therefore,

$$
a(\eta)
=======

-\log(1-\phi).
$$

Substituting

$$
\phi=\frac{1}{1+e^{-\eta}},
$$

gives

$$
a(\eta)
=======

-\log\left(
1-\frac{1}{1+e^{-\eta}}
\right)
=======

\log(1+e^\eta).
$$

Thus, the Bernoulli distribution is an exponential-family distribution with

$$
\boxed{
\eta=\log\left(\frac{\phi}{1-\phi}\right)
}
$$

and

$$
\boxed{
a(\eta)=\log(1+e^\eta)
}.
$$

### Gaussian Distribution (Fixed Variance)

For a Gaussian distribution with fixed variance $\sigma^2$,

$$
y\sim\mathcal N(\mu,\sigma^2),
$$

the density is

$$
P(y;\mu)
========

\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(
-\frac{(y-\mu)^2}{2\sigma^2}
\right).
$$

Expanding the exponent,

$$
P(y;\mu)
========

\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(
\frac{\mu}{\sigma^2}y
---------------------

## \frac{\mu^2}{2\sigma^2}

\frac{y^2}{2\sigma^2}
\right).
$$

Comparing with

$$
P(y;\eta)
=========

b(y)\exp\left(\eta T(y)-a(\eta)\right),
$$

we can identify

$$
T(y)=y,
$$

$$
\eta=\frac{\mu}{\sigma^2},
$$

and

$$
a(\eta)=\frac{\sigma^2\eta^2}{2}.
$$

The terms that depend only on $y$ are absorbed into $b(y)$.
