# Probability
## The best probability course：[Statistics 110: Probability-YouTube](https://www.youtube.com/watch?v=KbB0FjPg0mw&list=EC2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo)
## Course Catalog
<details>
<summary>Harvard Stat 110: Probability — Lecture Catalog (click to expand)</summary>


| Lecture | Title                                      | YouTube Link (Direct) |
|---------|--------------------------------------------|-----------------------|
| 1      | Probability and Counting                   | [Watch →](https://www.youtube.com/watch?v=KbB0FjPg0mw) |
| 2      | Story Proofs, Axioms of Probability        | [Watch →](https://www.youtube.com/watch?v=FJd_1H3rZGg) |
| 3      | Birthday Problem, Properties of Probability| [Watch →](https://www.youtube.com/watch?v=LZ5Wergp_PA) |
| 4      | Conditional Probability                    |  |
| 5      | Conditioning Continued, Law of Total Probability |  |
| 6      | Monty Hall, Simpson's Paradox              |  |
| 7      | Gambler's Ruin and Random Variables        |  |
| 8      | Random Variables and Their Distributions   |  |
| 9      | Expectation, Indicator Random Variables, Linearity | |
| 10     | Expectation Continued                      |  |
| 11     | The Poisson distribution                   |  |
| 12     | Discrete vs. Continuous, the Uniform       | ... |
| 13     | Normal distribution                        | ... |
| 14     | Location, Scale, and LOTUS                 | ... |
| 15     | Midterm Review                             | ... |
| 16     | Exponential Distribution                   | ... |
| 17     | Moment Generating Functions                | ... |
| 18     | MGFs Continued                             | ... |
| 19     | Joint, Conditional, and Marginal Distributions | ... |
| 20     | Multinomial and Cauchy                     | ... |
| 21     | Covariance and Correlation                 | ... |
| 22     | Transformations and Convolutions           | ... |
| 23     | Beta distribution                          | ... |
| 24     | Gamma distribution and Poisson process     | ... |
| 25     | Order Statistics and Conditional Expectation | ... |
| 26     | Conditional Expectation Continued          | ... |
| 27     | Conditional Expectation given an R.V.      | ... |
| 28     | Inequalities                               | ... |
| 29     | Law of Large Numbers and Central Limit Theorem | ... |
| 30     | Chi-Square, Student-t, Multivariate Normal | ... |
| 31     | Markov Chains                              | ... |
| 32     | Markov Chains Continued                    | ... |
| 33     | Markov Chains Continued Further            | ... |
| 34     | A Look Ahead                               | ... |

</details>

## Mathematical symbols and abbreviations related to probability theory
## Probability Theory: Common Symbols & Abbreviations

<details >
<summary>Basic Probability Symbols (click to collapse/expand)</summary>

| Recommended Markdown          | Rendered (should appear as math)     | Meaning / Name                              | Example / Notes                              |
|-------------------------------|--------------------------------------|---------------------------------------------|----------------------------------------------|
| $`P(A)`$                      | $`P(A)`$                             | Probability of event A                      | $`P(A) = 0.3`$                               |
| $`P(A \cap B)`$ or $`P(A,B)`$ | $`P(A \cap B)`$                      | Joint / intersection probability            | $`P(A \cap B) = P(A)P(B \mid A)`$            |
| $`P(A \cup B)`$               | $`P(A \cup B)`$                      | Union probability                           | $`P(A \cup B) = P(A) + P(B) - P(A \cap B)`$  |
| $`P(A^c)`$ or $`\bar{A}`$     | $`P(A^c)`$                           | Complement probability                      | $`P(A^c) = 1 - P(A)`$                        |
| $`P(A \mid B)`$               | $`P(A \mid B)`$                      | Conditional probability                     | $`P(A \mid B) = \frac{P(A \cap B)}{P(B)}`$   |
| $`\mathbb{P}`$                | $`\mathbb{P}`$                       | Probability measure (measure-theoretic)     | Often used in advanced / axiomatic contexts  |

</details>

<details>
<summary>Random Variables & Moments</summary>

| Recommended Markdown                   | Rendered                                 | Meaning                                      | Example / Notes                                  |
|----------------------------------------|------------------------------------------|----------------------------------------------|--------------------------------------------------|
| $`X, Y`$                               | $`X, Y`$                                 | Random variables                             | Uppercase usually denotes r.v.                   |
| $`x, y`$                               | $`x, y`$                                 | Realization / value of a r.v.                | Lowercase for observed values                    |
| $`X \sim \text{Dist}(\cdot)`$          | $`X \sim \text{Dist}(\cdot)`$            | X is distributed as ...                      | Core notation                                    |
| $`\sim_{\text{i.i.d.}}`$               | $`\sim_{\text{i.i.d.}}`$                 | Independent and identically distributed      | Very common in statistics                        |
| $`E[X]`$ or $`\mathbb{E}[X]`$          | $`E[X]`$ or $`\mathbb{E}[X]`$            | Expectation / mean                           | $`E[X] = \mu`$                                   |
| $`\mathrm{Var}(X)`$ or $`\sigma_X^2`$  | $`\mathrm{Var}(X)`$                      | Variance                                     | $`\mathrm{Var}(X) = E[(X - \mu)^2]`$             |
| $`\mathrm{SD}(X)`$ or $`\sigma_X`$     | $`\mathrm{SD}(X)`$                       | Standard deviation                           | Square root of variance                          |
| $`\mathrm{Cov}(X,Y)`$                  | $`\mathrm{Cov}(X,Y)`$                    | Covariance                                   | $`\mathrm{Cov}(X,Y) = E[XY] - E[X]E[Y]`$         |
| $`\rho_{X,Y}`$ or $`\mathrm{Corr}(X,Y)`$ | $`\rho_{X,Y}`$                         | Correlation coefficient                      | $`-1 \leq \rho \leq 1`$                          |

</details>

<details>
<summary>Common Distributions & Notation</summary>

| Recommended Markdown                   | Rendered                                 | Distribution / Name                          | Typical Parameters                               |
|----------------------------------------|------------------------------------------|----------------------------------------------|--------------------------------------------------|
| $`X \sim \text{Bern}(p)`$              | $`X \sim \text{Bern}(p)`$                | Bernoulli                                    | $`p \in [0,1]`$                                  |
| $`X \sim \text{Bin}(n,p)`$             | $`X \sim \text{Bin}(n,p)`$               | Binomial                                     | $`n`$ trials, success prob $`p`$                 |
| $`X \sim \text{Pois}(\lambda)`$        | $`X \sim \text{Pois}(\lambda)`$          | Poisson                                      | $`\lambda > 0`$ (rate)                           |
| $`X \sim U[a,b]`$ or $`\text{Unif}[a,b]`$ | $`X \sim U[a,b]`$                     | Uniform (continuous)                         | Interval $`[a,b]`$                               |
| $`X \sim \text{Exp}(\lambda)`$         | $`X \sim \text{Exp}(\lambda)`$           | Exponential                                  | Rate $`\lambda`$ or mean $`1/\lambda`$           |
| $`X \sim N(\mu, \sigma^2)`$ or $`\mathcal{N}(\mu, \sigma^2)`$ | $`X \sim N(\mu, \sigma^2)`$ | Normal / Gaussian                            | Mean $`\mu`$, variance $`\sigma^2`$              |

</details>

<details>
<summary>Key Abbreviations (Text & Math)</summary>

| Abbreviation   | Full Term                                      | Meaning (English / 中文)              | Common Usage Example                             |
|----------------|------------------------------------------------|---------------------------------------|--------------------------------------------------|
| **r.v.**       | random variable                                | 随机变量                              | "Let X be a r.v."                                |
| **i.i.d.**     | independent and identically distributed        | 独立同分布                            | $`X_1, \dots, X_n \sim_{\text{i.i.d.}} N(0,1)`$  |
| **a.s.**       | almost surely                                  | 几乎处处 / 几乎肯定                   | $`X_n \to X$` a.s.                               |
| **a.e.**       | almost everywhere                              | 几乎处处 (measure theory)             | Equivalent to a.s. in most contexts              |
| **cdf** / **CDF** | cumulative distribution function            | 累积分布函数                          | $`F_X(x) = P(X \leq x)`$                         |
| **pdf** / **PDF** | probability density function                 | 概率密度函数 (continuous)             | $`f_X(x)`$                                       |
| **pmf** / **PMF** | probability mass function                    | 概率质量函数 (discrete)               | $`p_X(x) = P(X = x)`$                            |
| **mgf**        | moment generating function                     | 矩母函数 / 矩生成函数                 | $`M_X(t) = E[e^{tX}]`$                           |
| **pgf**        | probability generating function                | 概率生成函数                          | $`G(s) = E[s^X]`$ (non-negative integer valued)  |
| **ch.f.**      | characteristic function                        | 特征函数                              | $`\phi_X(t) = E[e^{itX}]`$                       |
| **LLN**        | Law of Large Numbers                           | 大数定律                              | Weak / strong versions                           |
| **CLT**        | Central Limit Theorem                          | 中心极限定理                          | Normal approximation                             |

</details>

## Key Laws and Theorems in Probability Theory

### 1. Kolmogorov's Axioms (1933) — The Foundation of Probability
Any probability measure P must satisfy:
- Non-negativity: $`P(A) \geq 0`$ for any event A  
- Normalization: $`P(\Omega) = 1`$ (sample space has probability 1)  
- Countable additivity: For disjoint events $`A_1, A_2, \dots`$,  
  $`P\left( \bigcup_{i=1}^\infty A_i \right) = \sum_{i=1}^\infty P(A_i)`$

Immediate consequences:
- $`P(\emptyset) = 0`$
- $`P(A^c) = 1 - P(A)`$ (complement rule)

### 2. Addition Rule (Inclusion–Exclusion Principle)
- Two events: $`P(A \cup B) = P(A) + P(B) - P(A \cap B)`$  
- Mutually exclusive: $`P(A \cup B) = P(A) + P(B)`$ if $`A \cap B = \emptyset`$  
- Three events:  
  $`P(A \cup B \cup C) = P(A)+P(B)+P(C) - P(A\cap B)-P(A\cap C)-P(B\cap C) + P(A\cap B\cap C)`$

### 3. Multiplication Rule
- General: $`P(A \cap B) = P(A) \cdot P(B \mid A) = P(B) \cdot P(A \mid B)`$  
- Independent events: $`P(A \cap B) = P(A) \cdot P(B)`$ if A and B are independent

### 4. Law of Total Probability
If $`\{B_1, \dots, B_n\}`$ is a partition of the sample space (mutually exclusive and exhaustive), then  
$`P(A) = \sum_{i=1}^n P(A \mid B_i) P(B_i)`$

Common special case (two parts):  
$`P(A) = P(A \mid B)P(B) + P(A \mid B^c)P(B^c)`$

### 5. Bayes' Theorem
$`P(B \mid A) = \frac{P(A \mid B) P(B)}{P(A)}`$

With total probability in the denominator:  
$`P(B_j \mid A) = \frac{P(A \mid B_j) P(B_j)}{\sum_i P(A \mid B_i) P(B_i)}`$

→ Core of Bayesian inference, diagnostic testing, updating beliefs, spam filtering, etc.

### 6. Law of Iterated Expectation (Tower Property)
$`E[X] = E[E[X \mid Y]]`$

More generally: $`E[E[X \mid Y,Z]] = E[X \mid Z]`$  
→ Extremely useful for computing expectations by conditioning

### 7. Law of Large Numbers (LLN)
Sample average converges to the true expectation.

- **Weak LLN** (convergence in probability):  
  If $`X_i`$ are i.i.d. with $`E[X_i] = \mu`$, then  
  $`\bar{X}_n = \frac{1}{n}\sum X_i \xrightarrow{P} \mu`$

- **Strong LLN** (almost sure convergence):  
  If $`E[|X_i|] < \infty`$, then $`\bar{X}_n \to \mu`$ almost surely

→ Mathematical justification that "frequencies approach probabilities"

### 8. Central Limit Theorem (CLT)
The cornerstone of modern statistics.

If $`X_i`$ are i.i.d. with finite mean $`\mu`$ and variance $`\sigma^2 > 0`$, then  
$`Z_n = \frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} N(0,1)`$ as $`n \to \infty`$

→ Explains the ubiquity of the normal distribution in real-world data

### 9. Fundamental Concentration Inequalities
- **Markov's Inequality**: For non-negative X and a > 0,  
  $`P(X \geq a) \leq \frac{E[X]}{a}`$

- **Chebyshev's Inequality**: For any X with finite variance,  
  $`P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}`$ (k > 0)

- **Jensen's Inequality** (for convex f):  
  $`f(E[X]) \leq E[f(X)]`$

### Priority Ranking (for learning / exams / applications)

| Priority | Law / Theorem                  | Why it matters most                                 | Typical applications                     |
|----------|--------------------------------|-----------------------------------------------------|------------------------------------------|
| ★★★★★    | Bayes' Theorem + Total Probability | Foundation of inference and updating beliefs        | Diagnostics, ML, A/B testing             |
| ★★★★★    | LLN + CLT                      | Basis of statistical inference and normal approximation | Confidence intervals, hypothesis testing |
| ★★★★     | Addition / Multiplication Rules | Everyday probability calculations                   | Any basic probability problem            |
| ★★★      | Markov / Chebyshev             | Proving bounds, tail probabilities                  | Algorithm analysis, concentration bounds |
