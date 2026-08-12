# Logistic Regression

## Hypothesis Function

We want the model output to satisfy:

<p align="center">
<strong>
h<sub>θ</sub>(x) ∈ [0, 1]
</strong>
</p>

This allows the output of the model to be interpreted as a probability.

The hypothesis function is:

<p align="center">
<strong>
h<sub>θ</sub>(x) = g(θᵀx) = 1 / (1 + e<sup>−θᵀx</sup>)
</strong>
</p>

---

## Sigmoid Function

The sigmoid function is:

<p align="center">
<strong>
g(z) = 1 / (1 + e<sup>−z</sup>)
</strong>
</p>

It maps any real-valued input to the interval `(0, 1)`.

---

## Probability of Classification

For binary classification:

<p align="center">
<strong>
y ∈ {0, 1}
</strong>
</p>

The probability that `y = 1` is:

<p align="center">
<strong>
P(y = 1 | x; θ) = h<sub>θ</sub>(x)
</strong>
</p>

The probability that `y = 0` is:

<p align="center">
<strong>
P(y = 0 | x; θ) = 1 − h<sub>θ</sub>(x)
</strong>
</p>

These can be combined into one expression:

<p align="center">
<strong>
P(y | x; θ) = h<sub>θ</sub>(x)<sup>y</sup>
(1 − h<sub>θ</sub>(x))<sup>1−y</sup>
</strong>
</p>

---

# Maximum Likelihood Estimation

Given `m` training examples, the likelihood function is:

<p align="center">
<strong>
L(θ) = ∏<sub>i=1</sub><sup>m</sup> P(y<sup>(i)</sup> | x<sup>(i)</sup>; θ)
</strong>
</p>

Using the Bernoulli model:

<p align="center">
<strong>
L(θ) =
∏<sub>i=1</sub><sup>m</sup>
h<sub>θ</sub>(x<sup>(i)</sup>)<sup>y<sup>(i)</sup></sup>
(1 − h<sub>θ</sub>(x<sup>(i)</sup>))<sup>1−y<sup>(i)</sup></sup>
</strong>
</p>

We choose the parameters `θ` that maximize the likelihood.

---

# Log-Likelihood

Taking the logarithm:

<p align="center">
<strong>
ℓ(θ) = log L(θ)
</strong>
</p>

Therefore:

<p align="center">
<strong>
ℓ(θ) =
Σ<sub>i=1</sub><sup>m</sup>
[
y<sup>(i)</sup> log h<sub>θ</sub>(x<sup>(i)</sup>)
+
(1 − y<sup>(i)</sup>) log(1 − h<sub>θ</sub>(x<sup>(i)</sup>))
]
</strong>
</p>

We choose `θ` to maximize `ℓ(θ)`.

---

# Algorithm Used to Maximize the Log-Likelihood

## Batch Gradient Ascent

The general gradient-ascent update is:

<p align="center">
<strong>
θ<sub>j</sub> ← θ<sub>j</sub> +
α ∂ℓ(θ) / ∂θ<sub>j</sub>
</strong>
</p>

For logistic regression:

<p align="center">
<strong>
θ<sub>j</sub> ← θ<sub>j</sub> +
α Σ<sub>i=1</sub><sup>m</sup>
(y<sup>(i)</sup> − h<sub>θ</sub>(x<sup>(i)</sup>))
x<sub>j</sub><sup>(i)</sup>
</strong>
</p>

In vector form:

<p align="center">
<strong>
θ ← θ +
α Σ<sub>i=1</sub><sup>m</sup>
(y<sup>(i)</sup> − h<sub>θ</sub>(x<sup>(i)</sup>))
x<sup>(i)</sup>
</strong>
</p>

---

# Newton's Method

Newton's method can be used to find the optimum more quickly than gradient ascent.

For a one-dimensional function:

<p align="center">
<strong>
θ<sup>(t+1)</sup> =
θ<sup>(t)</sup> −
f′(θ<sup>(t)</sup>) / f″(θ<sup>(t)</sup>)
</strong>
</p>

For multiple parameters, the Hessian matrix is used:

<p align="center">
<strong>
θ<sup>(t+1)</sup> =
θ<sup>(t)</sup> −
H<sup>−1</sup> ∇J(θ<sup>(t)</sup>)
</strong>
</p>

A common compact form is:

<p align="center">
<strong>
θ ← θ − H<sup>−1</sup> ∇J(θ)
</strong>
</p>

The Hessian matrix contains second-order partial derivatives:

<p align="center">
<strong>
H<sub>ij</sub> =
∂²J(θ) / (∂θ<sub>i</sub> ∂θ<sub>j</sub>)
</strong>
</p>

---

# Summary

| Topic                   | Key Idea                                                           |
| ----------------------- | ------------------------------------------------------------------ |
| **Sigmoid**             | Converts a real number into a value between 0 and 1                |
| **Logistic Regression** | Uses the sigmoid output as a probability                           |
| **Maximum Likelihood**  | Chooses `θ` to maximize the probability of the training data       |
| **Log-Likelihood**      | Logarithm of the likelihood, making products easier to work with   |
| **Gradient Ascent**     | Iteratively updates `θ` in the direction that increases likelihood |
| **Newton's Method**     | Uses first- and second-order derivatives for faster optimization   |
