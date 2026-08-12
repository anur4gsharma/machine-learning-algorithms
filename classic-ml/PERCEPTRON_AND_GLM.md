# Perceptron and Generalized Linear Models

## Perceptron

### Logistic Regression vs. Perceptron

| Logistic / Sigmoid Function                                      | Perceptron / Step Function                          |
| ---------------------------------------------------------------- | --------------------------------------------------- |
| **Sigmoid:**<br>g(z) = 1 / (1 + e⁻ᶻ)                             | **Step:**<br>g(z) = 1 if z ≥ 0<br>g(z) = 0 if z < 0 |
| **Hypothesis:**<br>h<sub>θ</sub>(x) = 1 / (1 + e<sup>−θᵀx</sup>) | **Hypothesis:**<br>h<sub>θ</sub>(x) = g(θᵀx)        |
| Output is a probability between 0 and 1.                         | Output is either 0 or 1.                            |

---

### Batch Gradient Ascent for the Perceptron

The parameter update is:

<p align="center">
<strong>θ<sub>j</sub> ← θ<sub>j</sub> + α Σ<sub>i=0</sub><sup>m</sup> (y<sup>(i)</sup> − h<sub>θ</sub>(x<sup>(i)</sup>))x<sub>j</sub><sup>(i)</sup></strong>
</p>

Where:

* **α** = learning rate
* **y<sup>(i)</sup>** = true label
* **h<sub>θ</sub>(x<sup>(i)</sup>)** = predicted label
* **x<sub>j</sub><sup>(i)</sup>** = feature `j` of training example `i`

For the perceptron:

* If the prediction is **correct**, then the error is **0**, so there is no update.
* If the prediction is **incorrect**, then the error is either **+1** or **−1**.

---

# Exponential Families

A probability distribution belongs to the **exponential family** if it can be written in the form:

<p align="center">
<strong>P(y; η) = b(y) exp(ηᵀT(y) − a(η))</strong>
</p>

### Components

| Symbol   | Meaning                |
| -------- | ---------------------- |
| **y**    | Observed data          |
| **η**    | Natural parameter      |
| **T(y)** | Sufficient statistic   |
| **b(y)** | Base measure           |
| **a(η)** | Log-partition function |

---

# Examples

## Bernoulli Distribution

For binary data:

<p align="center"><strong>y ∈ {0, 1}</strong></p>

Let:

<p align="center"><strong>φ = P(y = 1)</strong></p>

The Bernoulli probability mass function is:

<p align="center">
<strong>P(y; φ) = φ<sup>y</sup>(1 − φ)<sup>1−y</sup></strong>
</p>

Rewrite it using an exponential:

<p align="center">
<strong>
P(y; φ) = exp(log(φ<sup>y</sup>(1 − φ)<sup>1−y</sup>))
</strong>
</p>

Using logarithm rules:

<p align="center">
<strong>
P(y; φ) = exp(y log φ + (1 − y) log(1 − φ))
</strong>
</p>

Expand the expression:

<p align="center">
<strong>
P(y; φ) = exp(y log φ − y log(1 − φ) + log(1 − φ))
</strong>
</p>

Therefore:

<p align="center">
<strong>
P(y; φ) = exp(y log(φ / (1 − φ)) + log(1 − φ))
</strong>
</p>

Compare this with the exponential-family form:

<p align="center">
<strong>P(y; η) = b(y) exp(ηT(y) − a(η))</strong>
</p>

We can identify:

<p align="center"><strong>b(y) = 1</strong></p>

<p align="center"><strong>T(y) = y</strong></p>

<p align="center">
<strong>
η = log(φ / (1 − φ))
</strong>
</p>

### Solving for φ

Starting from:

<p align="center">
<strong>
η = log(φ / (1 − φ))
</strong>
</p>

Exponentiating both sides:

<p align="center">
<strong>
e<sup>η</sup> = φ / (1 − φ)
</strong>
</p>

Rearranging:

<p align="center">
<strong>
φ = e<sup>η</sup> / (1 + e<sup>η</sup>)
</strong>
</p>

Therefore:

<p align="center">
<strong>
φ = 1 / (1 + e<sup>−η</sup>)
</strong>
</p>

### Finding a(η)

From the exponential-family form:

<p align="center">
<strong>
−a(η) = log(1 − φ)
</strong>
</p>

Therefore:

<p align="center">
<strong>
a(η) = −log(1 − φ)
</strong>
</p>

Substituting:

<p align="center">
<strong>
φ = 1 / (1 + e<sup>−η</sup>)
</strong>
</p>

gives:

<p align="center">
<strong>
a(η) = −log(1 − 1 / (1 + e<sup>−η</sup>))
</strong>
</p>

which simplifies to:

<p align="center">
<strong>
a(η) = log(1 + e<sup>η</sup>)
</strong>
</p>

### Final Bernoulli Exponential-Family Form

| Component | Value                    |
| --------- | ------------------------ |
| **b(y)**  | 1                        |
| **T(y)**  | y                        |
| **η**     | log(φ / (1 − φ))         |
| **φ**     | 1 / (1 + e<sup>−η</sup>) |
| **a(η)**  | log(1 + e<sup>η</sup>)   |

---

## Gaussian Distribution (Fixed Variance)

Consider:

<p align="center">
<strong>
y ~ N(μ, σ²)
</strong>
</p>

The Gaussian probability density function is:

<p align="center">
<strong>
P(y; μ) = 1 / √(2πσ²) × exp(−(y − μ)² / (2σ²))
</strong>
</p>

Expand the squared term:

<p align="center">
<strong>
(y − μ)² = y² − 2μy + μ²
</strong>
</p>

Substituting:

<p align="center">
<strong>
P(y; μ) = 1 / √(2πσ²) × exp((μ/σ²)y − μ²/(2σ²) − y²/(2σ²))
</strong>
</p>

Compare this with:

<p align="center">
<strong>
P(y; η) = b(y) exp(ηT(y) − a(η))
</strong>
</p>

We can identify:

<p align="center"><strong>T(y) = y</strong></p>

<p align="center">
<strong>
η = μ / σ²
</strong>
</p>

Since:

<p align="center">
<strong>
μ = ησ²
</strong>
</p>

we get:

<p align="center">
<strong>
a(η) = σ²η² / 2
</strong>
</p>

The terms that depend only on `y` are included in the base measure:

<p align="center">
<strong>
b(y) = 1 / √(2πσ²) × exp(−y² / (2σ²))
</strong>
</p>

### Final Gaussian Exponential-Family Form

| Component | Value                          |
| --------- | ------------------------------ |
| **b(y)**  | 1 / √(2πσ²) × exp(−y² / (2σ²)) |
| **T(y)**  | y                              |
| **η**     | μ / σ²                         |
| **a(η)**  | σ²η² / 2                       |
