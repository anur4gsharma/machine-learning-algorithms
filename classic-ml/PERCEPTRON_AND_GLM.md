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

A probability distribution belongs to the **exponential family** if it can be written in the form

$$
P(y;\eta)
=========

b(y)\exp\left(\eta^T T(y)-a(\eta)\right).
$$

Here:

* $y$ is the observed data.
* $\eta$ is the **natural parameter**.
* $T(y)$ is the **sufficient statistic**.
* $b(y)$ is the **base measure**.
* $a(\eta)$ is the **log-partition function**.

---

## Examples

### Bernoulli Distribution

For binary data,

$$
y\in{0,1}.
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

We can rewrite this using an exponential:

$$
P(y;\phi)
=========

\exp\left(
\log\left(
\phi^y(1-\phi)^{1-y}
\right)
\right).
$$

Using logarithm rules,

$$
P(y;\phi)
=========

\exp\left(
y\log\phi
+
(1-y)\log(1-\phi)
\right).
$$

Expanding,

$$
P(y;\phi)
=========

\exp\left(
y\log\phi
---------

y\log(1-\phi)
+
\log(1-\phi)
\right).
$$

Therefore,

$$
P(y;\phi)
=========

\exp\left(
y\log\left(\frac{\phi}{1-\phi}\right)
+
\log(1-\phi)
\right).
$$

Now compare this with the exponential-family form:

$$
P(y;\eta)
=========

b(y)\exp\left(
\eta T(y)-a(\eta)
\right).
$$

We can identify the individual components as

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

\log\left(
\frac{\phi}{1-\phi}
\right).
$$

### Solving for $\phi$

Starting with

$$
\eta
====

\log\left(
\frac{\phi}{1-\phi}
\right),
$$

exponentiating both sides gives

$$
e^\eta
======

\frac{\phi}{1-\phi}.
$$

Therefore,

$$
e^\eta(1-\phi)=\phi.
$$

Rearranging,

$$
e^\eta
======

\phi(1+e^\eta).
$$

Hence,

$$
\phi
====

# \frac{e^\eta}{1+e^\eta}

\frac{1}{1+e^{-\eta}}.
$$

### Finding $a(\eta)$

From the exponential-family form,

$$
-a(\eta)=\log(1-\phi).
$$

Therefore,

$$
a(\eta)
=======

-\log(1-\phi).
$$

Substituting

$$
\phi
====

\frac{1}{1+e^{-\eta}},
$$

we get

$$
a(\eta)
=======

-\log\left(
1-\frac{1}{1+e^{-\eta}}
\right).
$$

Simplifying,

$$
a(\eta)
=======

\log(1+e^\eta).
$$

Thus, the Bernoulli distribution can be written in exponential-family form with

$$
\boxed{
b(y)=1
}
$$

$$
\boxed{
T(y)=y
}
$$

$$
\boxed{
\eta
====

\log\left(
\frac{\phi}{1-\phi}
\right)
}
$$

and

$$
\boxed{
a(\eta)=\log(1+e^\eta)
}.
$$

---

### Gaussian Distribution (Fixed Variance)

Consider a Gaussian random variable with fixed variance $\sigma^2$:

$$
y\sim\mathcal N(\mu,\sigma^2).
$$

Its probability density function is

$$
P(y;\mu)
========

\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(
-\frac{(y-\mu)^2}{2\sigma^2}
\right).
$$

Expand the squared term:

$$
(y-\mu)^2
=========

y^2-2\mu y+\mu^2.
$$

Substituting this into the density,

$$
P(y;\mu)
========

\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(
-\frac{y^2-2\mu y+\mu^2}{2\sigma^2}
\right).
$$

Separating the terms,

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

Now compare this with

$$
P(y;\eta)
=========

b(y)
\exp\left(
\eta T(y)-a(\eta)
\right).
$$

We can identify

$$
T(y)=y,
$$

and

$$
\eta
====

\frac{\mu}{\sigma^2}.
$$

Since

$$
\mu=\eta\sigma^2,
$$

we have

$$
\frac{\mu^2}{2\sigma^2}
=======================

\frac{\eta^2\sigma^2}{2}.
$$

Therefore,

$$
a(\eta)
=======

\frac{\sigma^2\eta^2}{2}.
$$

The terms that depend only on $y$ are absorbed into $b(y)$:

$$
b(y)
====

\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(
-\frac{y^2}{2\sigma^2}
\right).
$$

Hence, for a Gaussian distribution with fixed variance,

$$
\boxed{
T(y)=y
}
$$

$$
\boxed{
\eta=\frac{\mu}{\sigma^2}
}
$$

and

$$
\boxed{
a(\eta)=\frac{\sigma^2\eta^2}{2}
}.
$$
