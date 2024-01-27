# Internal algorithms

This document contains internal algorithm documentation for Sleipnir.

## Reverse accumulation automatic differentiation

In reverse accumulation AD, the dependent variable to be differentiated is fixed and the derivative is computed with respect to each subexpression recursively. In a pen-and-paper calculation, the derivative of the outer functions is repeatedly substituted in the chain rule:

(∂y/∂x) = (∂y/∂w₁) ⋅ (∂w₁/∂x) = ((∂y/∂w₂) ⋅ (∂w₂/∂w₁)) ⋅ (∂w₁/∂x) = ...

In reverse accumulation, the quantity of interest is the adjoint, denoted with a bar (w̄); it is a derivative of a chosen dependent variable with respect to a subexpression w: ∂y/∂w.

Given the expression f(x₁,x₂) = sin(x₁) + x₁x₂, the computational graph is:
@mermaid{reverse-autodiff}

The operations to compute the derivative:

w̄₅ = 1 (seed)<br>
w̄₄ = w̄₅(∂w₅/∂w₄) = w̄₅<br>
w̄₃ = w̄₅(∂w₅/∂w₃) = w̄₅<br>
w̄₂ = w̄₃(∂w₃/∂w₂) = w̄₃w₁<br>
w̄₁ = w̄₄(∂w₄/∂w₁) + w̄₃(∂w₃/∂w₁) = w̄₄cos(w₁) + w̄₃w₂

https://en.wikipedia.org/wiki/Automatic_differentiation#Beyond_forward_and_reverse_accumulation

## Interior-point method

Let f(x)ₖ be the cost function, cₑ(x)ₖ be the equality constraints, and cᵢ(x)ₖ be the inequality constraints. The Lagrangian of the optimization problem is

```
  L(x, s, y, z)ₖ = f(x)ₖ − yₖᵀcₑ(x)ₖ − zₖᵀ(cᵢ(x)ₖ − sₖ)
```

The Hessian of the Lagrangian is

```
  H(x)ₖ = ∇²ₓₓL(x, s, y, z)ₖ
```

The primal-dual barrier term Hessian Σ is defined as

```
  Σ = S⁻¹Z
```

where

```
      [s₁ 0 ⋯ 0 ]
  S = [0  ⋱   ⋮ ]
      [⋮    ⋱ 0 ]
      [0  ⋯ 0 sₘ]

      [z₁ 0 ⋯ 0 ]
  Z = [0  ⋱   ⋮ ]
      [⋮    ⋱ 0 ]
      [0  ⋯ 0 zₘ]
```

and where m is the number of inequality constraints.

Let f(x) = f(x)ₖ, H = H(x)ₖ, Aₑ = Aₑ(x)ₖ, and Aᵢ = Aᵢ(x)ₖ for clarity. We want to solve the following Newton-KKT system shown in equation (19.12) of [^1].

```
  [H    0  Aₑᵀ  Aᵢᵀ][ pₖˣ]    [∇f(x) − Aₑᵀy − Aᵢᵀz]
  [0    Σ   0   −I ][ pₖˢ] = −[     z − μS⁻¹e     ]
  [Aₑ   0   0    0 ][−pₖʸ]    [        cₑ         ]
  [Aᵢ  −I   0    0 ][−pₖᶻ]    [      cᵢ − s       ]
```

where e is a column vector of ones with a number of rows equal to the number of inequality constraints.

Solve the second row for pₖˢ.

```
  Σpₖˢ + pₖᶻ = μS⁻¹e − z
  Σpₖˢ = μS⁻¹e − z − pₖᶻ
  pₖˢ = μΣ⁻¹S⁻¹e − Σ⁻¹z − Σ⁻¹pₖᶻ
```

Substitute Σ = S⁻¹Z into the first two terms.

```
  pₖˢ = μ(S⁻¹Z)⁻¹S⁻¹e − (S⁻¹Z)⁻¹z − Σ⁻¹pₖᶻ
  pₖˢ = μZ⁻¹SS⁻¹e − Z⁻¹Sz − Σ⁻¹pₖᶻ
  pₖˢ = μZ⁻¹e − s − Σ⁻¹pₖᶻ
```

Substitute the explicit formula for pₖˢ into the fourth row and simplify.

```
  Aᵢpₖˣ − pₖˢ = s − cᵢ
  Aᵢpₖˣ − (μZ⁻¹e − s − Σ⁻¹pₖᶻ) = s − cᵢ
  Aᵢpₖˣ − μZ⁻¹e + s + Σ⁻¹pₖᶻ = s − cᵢ
  Aᵢpₖˣ + Σ⁻¹pₖᶻ = −cᵢ + μZ⁻¹e
```

Substitute the new second and fourth rows into the system.

```
  [H   0  Aₑᵀ  Aᵢᵀ ][ pₖˣ]    [∇f(x) − Aₑᵀy − Aᵢᵀz]
  [0   I   0    0  ][ pₖˢ] = −[−μZ⁻¹e + s + Σ⁻¹pₖᶻ]
  [Aₑ  0   0    0  ][−pₖʸ]    [        cₑ         ]
  [Aᵢ  0   0   −Σ⁻¹][−pₖᶻ]    [     cᵢ − μZ⁻¹e    ]
```

Eliminate the second row and column.

```
  [H   Aₑᵀ  Aᵢᵀ ][ pₖˣ]    [∇f(x) − Aₑᵀy − Aᵢᵀz]
  [Aₑ   0    0  ][−pₖʸ] = −[        cₑ         ]
  [Aᵢ   0   −Σ⁻¹][−pₖᶻ]    [    cᵢ − μZ⁻¹e     ]
```

Solve the third row for pₖᶻ.

```
  Aₑpₖˣ + Σ⁻¹pₖᶻ = −cᵢ + μZ⁻¹e
  pₖᶻ = −Σcᵢ + μS⁻¹e − ΣAᵢpₖˣ
```

Substitute the explicit formula for pₖᶻ into the first row.

```
  Hpₖˣ − Aₑᵀpₖʸ − Aᵢᵀpₖᶻ = −∇f(x) + Aₑᵀy + Aᵢᵀz
  Hpₖˣ − Aₑᵀpₖʸ − Aᵢᵀ(−Σcᵢ + μS⁻¹e − ΣAᵢpₖˣ) = −∇f(x) + Aₑᵀy + Aᵢᵀz
```

Expand and simplify.

```
  Hpₖˣ − Aₑᵀpₖʸ + AᵢᵀΣcᵢ − AᵢᵀμS⁻¹e + AᵢᵀΣAᵢpₖˣ = −∇f(x) + Aₑᵀy + Aᵢᵀz
  Hpₖˣ + AᵢᵀΣAᵢpₖˣ − Aₑᵀpₖʸ  = −∇f(x) + Aₑᵀy + AᵢᵀΣcᵢ + AᵢᵀμS⁻¹e + Aᵢᵀz
  (H + AᵢᵀΣAᵢ)pₖˣ − Aₑᵀpₖʸ = −∇f(x) + Aₑᵀy − Aᵢᵀ(Σcᵢ − μS⁻¹e − z)
  (H + AᵢᵀΣAᵢ)pₖˣ − Aₑᵀpₖʸ = −(∇f(x) − Aₑᵀy + Aᵢᵀ(Σcᵢ − μS⁻¹e − z))
```

Substitute the new first and third rows into the system.

```
  [H + AᵢᵀΣAᵢ   Aₑᵀ  0][ pₖˣ]    [∇f(x) − Aₑᵀy + Aᵢᵀ(Σcᵢ − μS⁻¹e − z)]
  [    Aₑ        0   0][−pₖʸ] = −[                cₑ                 ]
  [    0         0   I][−pₖᶻ]    [       −Σcᵢ + μS⁻¹e − ΣAᵢpₖˣ       ]
```

Eliminate the third row and column.

```
  [H + AᵢᵀΣAᵢ  Aₑᵀ][ pₖˣ] = −[∇f − Aₑᵀy + Aᵢᵀ(S⁻¹(Zcᵢ − μe) − z)]
  [    Aₑ       0 ][−pₖʸ]    [                cₑ                ]
```

This reduced 2x2 block system gives the iterates pₖˣ and pₖʸ with the iterates pₖᶻ and pₖˢ given by

```
  pₖᶻ = −Σcᵢ + μS⁻¹e − ΣAᵢpₖˣ
  pₖˢ = μZ⁻¹e − s − Σ⁻¹pₖᶻ
      = μZ⁻¹e − s − (S⁻¹Z)⁻¹pₖᶻ
      = μZ⁻¹e − s − Z⁻¹Spₖᶻ
```

The iterates are applied like so

```
  xₖ₊₁ = xₖ + αₖᵐᵃˣpₖˣ
  sₖ₊₁ = sₖ + αₖᵐᵃˣpₖˢ
  yₖ₊₁ = yₖ + αₖᶻpₖʸ
  zₖ₊₁ = zₖ + αₖᶻpₖᶻ
```

where αₖᵐᵃˣ and αₖᶻ are computed via the fraction-to-the-boundary rule shown in equations (15a) and (15b) of [^2].

```
  αₖᵐᵃˣ = max(α ∈ (0, 1] : sₖ + αpₖˢ ≥ (1−τⱼ)sₖ)
        = max(α ∈ (0, 1] : αpₖˢ ≥ −τⱼsₖ)
  αₖᶻ = max(α ∈ (0, 1] : zₖ + αpₖᶻ ≥ (1−τⱼ)zₖ)
      = max(α ∈ (0, 1] : αpₖᶻ ≥ −τⱼzₖ)
```

Section 6 of [^3] describes how to check for local infeasibility.

## Works cited

[^1]: Hinder, O. and Ye, Y. "A one-phase interior point method for nonconvex optimization", 2018. [https://arxiv.org/pdf/1801.03072.pdf](https://arxiv.org/pdf/1801.03072.pdf)

[^2]: Nocedal, J. and Wright, S. "Numerical Optimization", 2nd. ed., Ch. 19. Springer, 2006.

[^3]: Wächter, A. and Biegler, L. "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming", 2005. [http://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf](http://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf)

[^4]: Byrd, R. and Nocedal, J. and Waltz, R. "KNITRO: An Integrated Package for Nonlinear Optimization", 2005. [https://users.iems.northwestern.edu/~nocedal/PDFfiles/integrated.pdf](https://users.iems.northwestern.edu/~nocedal/PDFfiles/integrated.pdf)
