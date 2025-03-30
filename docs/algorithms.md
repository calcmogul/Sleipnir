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

## Unconstrained optimization

We want to solve the following optimization problem.

```
   min f(x)
    x
```

where f(x) is the cost function.

### Lagrangian

The Lagrangian of the problem is

```
  L(x) = f(x)
```

### Gradients of the Lagrangian

The gradients are

```
  ∇ₓL(x) = ∇f
```

The first-order necessary conditions for optimality are

```
  ∇f = 0
```

### Newton's method

Next, we'll apply Newton's method to the optimality conditions. Let H be ∂²L/∂x² and pˣ be the step for x.

```
  ∇ₓL(x + pˣ) ≈ ∇ₓL(x) + ∂²L/∂x²pˣ
  ∇ₓL(x) + Hpˣ = 0
  Hpˣ = −∇ₓL(x, y)
  Hpˣ = −(∇f)
```

### Final results

In summary, the following system gives the iterate pₖˣ.

```
  Hpˣ = −∇f(x)
```

The iterate is applied like so

```
  xₖ₊₁ = xₖ + pₖˣ
```

## Sequential quadratic programming

We want to solve the following optimization problem.

```
   min f(x)
    x
  s.t. cₑ(x) = 0
```

where f(x) is the cost function and cₑ(x) is the equality constraints.

### Lagrangian

The Lagrangian of the problem is

```
  L(x, y) = f(x) − yᵀcₑ(x)
```

### Gradients of the Lagrangian

The gradients are

```
  ∇ₓL(x, y) = ∇f − Aₑᵀy
  ∇_yL(x, y) = −cₑ
```

The first-order necessary conditions for optimality are

```
  ∇f − Aₑᵀy = 0
  −cₑ = 0
```

where Aₑ = ∂cₑ/∂x. We'll rearrange them for the primal-dual system.

```
  ∇f − Aₑᵀy = 0
  cₑ = 0
```

### Newton's method

Next, we'll apply Newton's method to the optimality conditions. Let H be ∂²L/∂x², pˣ be the step for x, and pʸ be the step for y.

```
  ∇ₓL(x + pˣ, y + pʸ) ≈ ∇ₓL(x, y) + ∂²L/∂x²pˣ + ∂²L/∂x∂ypʸ
  ∇ₓL(x, y) + Hpˣ − Aₑᵀpʸ = 0
  Hpˣ − Aₑᵀpʸ = −∇ₓL(x, y)
  Hpˣ − Aₑᵀpʸ = −(∇f − Aₑᵀy)
```
```
  ∇_yL(x + pˣ, y + pʸ) ≈ ∇_yL(x, y) + ∂²L/∂y∂xpˣ + ∂²L/∂y²pʸ
  ∇_yL(x, y) + Aₑpˣ = 0
  Aₑpˣ = −∇_yL(x, y)
  Aₑpˣ = −cₑ
```

### Matrix equation

Group them into a matrix equation.

```
  [H   −Aₑᵀ][pˣ] = −[∇f(x) − Aₑᵀy]
  [Aₑ   0  ][pʸ]    [     cₑ     ]
```

Invert pʸ.

```
  [H   Aₑᵀ][ pˣ] = −[∇f(x) − Aₑᵀy]
  [Aₑ   0 ][−pʸ]    [     cₑ     ]
```

### Final results

In summary, the reduced 2x2 block system gives the iterates pₖˣ and pₖʸ.

```
  [H   Aₑᵀ][ pˣ] = −[∇f(x) − Aₑᵀy]
  [Aₑ   0 ][−pʸ]    [     cₑ     ]
```

The iterates are applied like so

```
  xₖ₊₁ = xₖ + pₖˣ
  yₖ₊₁ = yₖ + pₖʸ
```

Section 6 of [^3] describes how to check for local infeasibility.

## Interior-point method

We want to solve the following optimization problem.

```
   min f(x)
    x
  s.t. cₑ(x) = 0
       cᵢ(x) ≥ 0
```

where f(x) is the cost function, cₑ(x) is the equality constraints, and cᵢ(x) is the inequality constraints. First, we'll reformulate the inequality constraints as equality constraints with slack variables.

```
   min f(x)
    x
  s.t. cₑ(x) = 0
       cᵢ(x) − s = 0
       s ≥ 0
```

To make this easier to solve, we'll reformulate it as the following barrier problem.

```
  min f(x) − μ Σ ln(sᵢ)
   x           i
  s.t. cₑ(x) = 0
       cᵢ(x) − s = 0
```

where μ is the barrier parameter. As μ → 0, the solution of the barrier problem approaches the solution of the original problem.

### Lagrangian

The Lagrangian of the barrier problem is

```
  L(x, s, y, z) = f(x) − μ Σ ln(sᵢ) − yᵀcₑ(x) − zᵀ(cᵢ(x) − s)
                           i
```

### Gradients of the Lagrangian

The gradients are

```
  ∇ₓL(x, s, y, z) = ∇f − Aₑᵀy − Aᵢᵀz
  ∇ₛL(x, s, y, z) = z − μS⁻¹e
  ∇_yL(x, s, y, z) = −cₑ
  ∇_zL(x, s, y, z) = −cᵢ + s
```

The first-order necessary conditions for optimality are

```
  ∇f − Aₑᵀy − Aᵢᵀz = 0
  z − μS⁻¹e = 0
  −cₑ = 0
  −cᵢ + s = 0
```

where Aₑ = ∂cₑ/∂x, Aᵢ = ∂cᵢ/∂x, S = diag(s), and e is a column vector of ones. We'll rearrange them for the primal-dual system.

```
  ∇f − Aₑᵀy − Aᵢᵀz = 0
  Sz − μe = 0
  cₑ = 0
  cᵢ − s = 0
```

### Newton's method

Next, we'll apply Newton's method to the optimality conditions. Let H be ∂²L/∂x², Σ be the primal-dual barrier term Hessian S⁻¹Z, pˣ be the step for x, pˢ be the step for s, pʸ be the step for y, and pᶻ be the step for z.

```
  ∇ₓL(x + pˣ, s + pˢ, y + pʸ, z + pᶻ)
    ≈ ∇ₓL(x, s, y, z) + ∂²L/∂x²pˣ + ∂²L/∂x∂spˢ + ∂²L/∂x∂ypʸ + ∂²L/∂x∂zpᶻ
  ∇ₓL(x, s, y, z) + Hpˣ − Aₑᵀpʸ − Aᵢᵀpᶻ = 0
  Hpˣ − Aₑᵀpʸ − Aᵢᵀpᶻ = −∇ₓL(x, s, y, z)
  Hpˣ − Aₑᵀpʸ − Aᵢᵀpᶻ = −(∇f − Aₑᵀy − Aᵢᵀz)
```
```
  ∇ₛL(x + pˣ, s + pˢ, y + pʸ, z + pᶻ)
    ≈ ∇ₛL(x, s, y, z) + ∂²L/∂s∂xpˣ + ∂²L/∂s²pˢ + ∂²L/∂s∂ypʸ + ∂²L/∂s∂zpᶻ
  ∇ₛL(x, s, y, z) + Zpˢ + Spᶻ = 0
  Zpˢ + Spᶻ = −∇ₛL(x, s, y, z)
  Zpˢ + Spᶻ = −(Sz − μe)
```
```
  ∇_yL(x + pˣ, s + pˢ, y + pʸ, z + pᶻ)
    ≈ ∇_yL(x, s, y, z) + ∂²L/∂y∂xpˣ + ∂²L/∂y∂spˢ + ∂²L/∂y²pʸ + ∂²L/∂y∂zpᶻ
  ∇_yL(x, s, y, z) + Aₑpˣ = 0
  Aₑpˣ = −∇_yL(x, s, y, z)
  Aₑpˣ = −cₑ
```
```
  ∇_zL(x + pˣ, s + pˢ, y + pʸ, z + pᶻ)
    ≈ ∇_zL(x, s, y, z) + ∂²L/∂z∂xpˣ + ∂²L/∂z∂spˢ + ∂²L/∂z∂ypʸ + ∂²L/∂z²pᶻ
  ∇_zL(x, s, y, z) + Aᵢpˣ − pˢ = 0
  Aᵢpˣ − pˢ = −∇_zL(x, s, y, z)
  Aᵢpˣ − pˢ = −(cᵢ − s)
```

### Matrix equation

Group them into a matrix equation.

```
  [H    0  −Aₑᵀ  −Aᵢᵀ][pˣ]    [∇f(x) − Aₑᵀy − Aᵢᵀz]
  [0    Z   0     S  ][pˢ] = −[      Sz − μe      ]
  [Aₑ   0   0     0  ][pʸ]    [        cₑ         ]
  [Aᵢ  −I   0     0  ][pᶻ]    [      cᵢ − s       ]
```

Invert pʸ and pᶻ.

```
  [H    0  Aₑᵀ  Aᵢᵀ][ pˣ]    [∇f(x) − Aₑᵀy − Aᵢᵀz]
  [0    Z   0    S ][ pˢ] = −[      Sz − μe      ]
  [Aₑ   0   0    0 ][−pʸ]    [        cₑ         ]
  [Aᵢ  −I   0    0 ][−pᶻ]    [      cᵢ − s       ]
```

Multiply the second row by S⁻¹ and replace S⁻¹Z with Σ.

```
  [H    0  Aₑᵀ  Aᵢᵀ][ pˣ]    [∇f(x) − Aₑᵀy − Aᵢᵀz]
  [0    Σ   0   −I ][ pˢ] = −[     z − μS⁻¹e     ]
  [Aₑ   0   0    0 ][−pʸ]    [        cₑ         ]
  [Aᵢ  −I   0    0 ][−pᶻ]    [      cᵢ − s       ]
```

Solve the second row for pˢ.

```
  Σpˢ + pᶻ = μS⁻¹e − z
  Σpˢ = μS⁻¹e − z − pᶻ
  pˢ = μΣ⁻¹S⁻¹e − Σ⁻¹z − Σ⁻¹pᶻ
```

Substitute Σ = S⁻¹Z into the first two terms.

```
  pˢ = μ(S⁻¹Z)⁻¹S⁻¹e − (S⁻¹Z)⁻¹z − Σ⁻¹pᶻ
  pˢ = μZ⁻¹SS⁻¹e − Z⁻¹Sz − Σ⁻¹pᶻ
  pˢ = μZ⁻¹e − s − Σ⁻¹pᶻ
```

Substitute the explicit formula for pˢ into the fourth row and simplify.

```
  Aᵢpˣ − pˢ = s − cᵢ
  Aᵢpˣ − (μZ⁻¹e − s − Σ⁻¹pᶻ) = s − cᵢ
  Aᵢpˣ − μZ⁻¹e + s + Σ⁻¹pᶻ = s − cᵢ
  Aᵢpˣ + Σ⁻¹pᶻ = −cᵢ + μZ⁻¹e
```

Substitute the new second and fourth rows into the system.

```
  [H   0  Aₑᵀ  Aᵢᵀ ][ pˣ]    [∇f(x) − Aₑᵀy − Aᵢᵀz]
  [0   I   0    0  ][ pˢ] = −[−μZ⁻¹e + s + Σ⁻¹pᶻ ]
  [Aₑ  0   0    0  ][−pʸ]    [        cₑ         ]
  [Aᵢ  0   0   −Σ⁻¹][−pᶻ]    [     cᵢ − μZ⁻¹e    ]
```

Eliminate the second row and column.

```
  [H   Aₑᵀ  Aᵢᵀ ][ pˣ]    [∇f(x) − Aₑᵀy − Aᵢᵀz]
  [Aₑ   0    0  ][−pʸ] = −[        cₑ         ]
  [Aᵢ   0   −Σ⁻¹][−pᶻ]    [    cᵢ − μZ⁻¹e     ]
```

Solve the third row for pᶻ.

```
  Aᵢpˣ + Σ⁻¹pᶻ = −cᵢ + μZ⁻¹e
  Σ⁻¹pᶻ = −cᵢ + μZ⁻¹e − Aᵢpˣ
  pᶻ = −Σcᵢ + μΣZ⁻¹e − ΣAᵢpˣ
  pᶻ = −Σcᵢ + μ(S⁻¹Z)Z⁻¹e − ΣAᵢpˣ
  pᶻ = −Σcᵢ + μS⁻¹e − ΣAᵢpˣ
```

Substitute the explicit formula for pᶻ into the first row.

```
  Hpˣ − Aₑᵀpʸ − Aᵢᵀpᶻ = −∇f(x) + Aₑᵀy + Aᵢᵀz
  Hpˣ − Aₑᵀpʸ − Aᵢᵀ(−Σcᵢ + μS⁻¹e − ΣAᵢpˣ) = −∇f(x) + Aₑᵀy + Aᵢᵀz
```

Expand and simplify.

```
  Hpˣ − Aₑᵀpʸ + AᵢᵀΣcᵢ − AᵢᵀμS⁻¹e + AᵢᵀΣAᵢpˣ = −∇f(x) + Aₑᵀy + Aᵢᵀz
  Hpˣ + AᵢᵀΣAᵢpˣ − Aₑᵀpʸ  = −∇f(x) + Aₑᵀy − AᵢᵀΣcᵢ + AᵢᵀμS⁻¹e + Aᵢᵀz
  (H + AᵢᵀΣAᵢ)pˣ − Aₑᵀpʸ = −∇f(x) + Aₑᵀy + Aᵢᵀ(−Σcᵢ + μS⁻¹e + z)
  (H + AᵢᵀΣAᵢ)pˣ − Aₑᵀpʸ = −(∇f(x) − Aₑᵀy − Aᵢᵀ(−Σcᵢ + μS⁻¹e + z))
```

Substitute the new first and third rows into the system.

```
  [H + AᵢᵀΣAᵢ   Aₑᵀ  0][ pˣ]    [∇f(x) − Aₑᵀy − Aᵢᵀ(−Σcᵢ + μS⁻¹e + z)]
  [    Aₑ        0   0][−pʸ] = −[                 cₑ                 ]
  [    0         0   I][−pᶻ]    [        −Σcᵢ + μS⁻¹e − ΣAᵢpˣ        ]
```

Eliminate the third row and column.

```
  [H + AᵢᵀΣAᵢ  Aₑᵀ][ pˣ] = −[∇f − Aₑᵀy − Aᵢᵀ(−Σcᵢ + μS⁻¹e + z)]
  [    Aₑ       0 ][−pʸ]    [               cₑ                ]
```

Expand and simplify pˢ.

```
  pˢ = μZ⁻¹e − s − Σ⁻¹pᶻ
  pˢ = μZ⁻¹e − s − (S⁻¹Z)⁻¹pᶻ
  pˢ = μZ⁻¹e − s − Z⁻¹Spᶻ
  pˢ = μZ⁻¹e − s − Z⁻¹S(−Σcᵢ + μS⁻¹e − ΣAᵢpˣ)
  pˢ = μZ⁻¹e − s − Z⁻¹S(−S⁻¹Zcᵢ + μS⁻¹e − S⁻¹ZAᵢpˣ)
  pˢ = μZ⁻¹e − s − Z⁻¹(−Zcᵢ + μe − ZAᵢpˣ)
  pˢ = μZ⁻¹e − s − (−cᵢ + μZ⁻¹e − Aᵢpˣ)
  pˢ = μZ⁻¹e − s + cᵢ − μZ⁻¹e + Aᵢpˣ
  pˢ = −s + cᵢ + Aᵢpˣ
  pˢ = cᵢ − s + Aᵢpˣ
```

### Final results

In summary, the reduced 2x2 block system gives the iterates pₖˣ and pₖʸ.

```
  [H + AᵢᵀΣAᵢ  Aₑᵀ][ pˣ] = −[∇f − Aₑᵀy − Aᵢᵀ(−Σcᵢ + μS⁻¹e + z)]
  [    Aₑ       0 ][−pʸ]    [               cₑ                ]
```

The iterates pˢ and pᶻ are given by

```
  pˢ = cᵢ − s + Aᵢpˣ
  pᶻ = −Σcᵢ + μS⁻¹e − ΣAᵢpˣ
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

[^1]: Nocedal, J. and Wright, S. "Numerical Optimization", 2nd. ed., Ch. 19. Springer, 2006.

[^2]: Wächter, A. and Biegler, L. "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming", 2005. [http://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf](http://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf)

[^3]: Byrd, R. and Nocedal, J. and Waltz, R. "KNITRO: An Integrated Package for Nonlinear Optimization", 2005. [https://users.iems.northwestern.edu/~nocedal/PDFfiles/integrated.pdf](https://users.iems.northwestern.edu/~nocedal/PDFfiles/integrated.pdf)

[^4]: Gu, C. and Zhu, D. "A Dwindling Filter Algorithm with a Modified Subproblem for Nonlinear Inequality Constrained Optimization", 2014. [https://sci-hub.st/10.1007/s11401-014-0826-z](https://sci-hub.st/10.1007/s11401-014-0826-z)

[^5]: Hinder, O. and Ye, Y. "A one-phase interior point method for nonconvex optimization", 2018. [https://arxiv.org/pdf/1801.03072.pdf](https://arxiv.org/pdf/1801.03072.pdf)

[^6]: Pougkakiotis, S., Gondzio, J. "Dynamic Non-diagonal Regularization in Interior Point Methods for Linear and Convex Quadratic Programming". J Optim Theory Appl 181, 905–945 (2019). https://doi.org/10.1007/s10957-019-01491-1
