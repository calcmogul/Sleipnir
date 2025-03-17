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

## Log-domain interior-point method

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

To ensure s ≥ 0 and z ≥ 0, make the following substitutions.

```
  s = √(μ)eᵛ
  z = √(μ)e⁻ᵛ
```
```
  ∇f − Aₑᵀy − Aᵢᵀ√(μ)e⁻ᵛ = 0
  cₑ = 0
  cᵢ − √(μ)eᵛ = 0

  ∇f − Aₑᵀy − √(μ)Aᵢᵀe⁻ᵛ = 0
  cₑ = 0
  cᵢ − √(μ)eᵛ = 0
```

The complementarity condition is now always satisfied, so it can be omitted.

### Newton's method

Next, we'll apply Newton's method to the optimality conditions. Let H be ∂²L/∂x², pˣ be the step for x, pʸ be the step for y, and pᵛ be the step for v.

```
  ∇ₓL(x + pˣ, y + pʸ, v + pᵛ)
    ≈ ∇ₓL(x, y, v) + ∂²L/∂x²pˣ + ∂²L/∂x∂ypʸ + ∂²L/∂x∂vpᵛ
  ∇ₓL(x, y, v) + Hpˣ − Aₑᵀpʸ + √(μ)Aᵢᵀe⁻ᵛ∘pᵛ = 0
  Hpˣ − Aₑᵀpʸ + √(μ)Aᵢᵀe⁻ᵛ∘pᵛ = −∇ₓL(x, y, v)
  Hpˣ − Aₑᵀpʸ + √(μ)Aᵢᵀe⁻ᵛ∘pᵛ = −(∇f − Aₑᵀy − √(μ)Aᵢᵀe⁻ᵛ)
```
```
  ∇_yL(x + pˣ, y + pʸ, v + pᵛ)
    ≈ ∇_yL(x, y, v) + ∂²L/∂y∂xpˣ + ∂²L/∂y²pʸ + ∂²L/∂y∂vpᵛ
  ∇_yL(x, y, v) + Aₑpˣ = 0
  Aₑpˣ = −∇_yL(x, y, v)
  Aₑpˣ = −cₑ
```
```
  ∇ᵥL(x + pˣ, y + pʸ, v + pᵛ)
    ≈ ∇ᵥL(x, y, v) + ∂²L/∂v∂xpˣ + ∂²L/∂v∂ypʸ + ∂²L/∂v²pᵛ
  ∇ᵥL(x, y, v) + Aᵢpˣ − √(μ)eᵛ∘pᵛ = 0
  Aᵢpˣ − √(μ)eᵛ∘pᵛ = −∇ᵥL(x, y, v)
  Aᵢpˣ − √(μ)eᵛ∘pᵛ = −(cᵢ − √(μ)eᵛ)
```

### Matrix equation

Group them into a matrix equation.

```
  [H   −Aₑᵀ  √(μ)Aᵢᵀe⁻ᵛ][pˣ]    [∇f − Aₑᵀy − √(μ)Aᵢᵀe⁻ᵛ]
  [Aₑ   0         0    ][pʸ] = −[          cₑ          ]
  [Aᵢ   0     −√(μ)eᵛ  ][pᵛ]    [     cᵢ − √(μ)eᵛ      ]
```

Invert pʸ.

```
  [H   Aₑᵀ  √(μ)Aᵢᵀe⁻ᵛ][ pˣ]    [∇f − Aₑᵀy − √(μ)Aᵢᵀe⁻ᵛ]
  [Aₑ   0       0     ][−pʸ] = −[          cₑ          ]
  [Aᵢ   0    −√(μ)eᵛ  ][ pᵛ]    [     cᵢ − √(μ)eᵛ      ]
```

Solve the third row for pᵛ.

```
  Aᵢpˣ − √(μ)eᵛ∘pᵛ = −cᵢ + √(μ)eᵛ
  −√(μ)eᵛ∘pᵛ = −Aᵢpˣ − cᵢ + √(μ)eᵛ
  pᵛ = 1/√(μ) Aᵢe⁻ᵛ∘pˣ + 1/√(μ) e⁻ᵛ∘cᵢ − e
  pᵛ = 1/√(μ) e⁻ᵛ∘(Aᵢpˣ + cᵢ) − e
```

Substitute the explicit formula for pᵛ into the first row.

```
  Hpˣ − Aₑᵀpʸ + √(μ)Aᵢᵀe⁻ᵛ∘pᵛ = −∇f + Aₑᵀy + √(μ)Aᵢᵀe⁻ᵛ
  Hpˣ − Aₑᵀpʸ + √(μ)Aᵢᵀe⁻ᵛ∘(1/√(μ) e⁻ᵛ∘(Aᵢpˣ + cᵢ) − e) = −∇f + Aₑᵀy − √(μ)Aᵢᵀe⁻ᵛ
```

Expand and simplify.

```
  Hpˣ − Aₑᵀpʸ + Aᵢᵀe⁻ᵛ∘(e⁻ᵛ∘(Aᵢpˣ + cᵢ) − √(μ)) = −∇f + Aₑᵀy − √(μ)Aᵢᵀe⁻ᵛ
  Hpˣ − Aₑᵀpʸ + Aᵢᵀe⁻²ᵛ∘(Aᵢpˣ + cᵢ) − √(μ)Aᵢᵀe⁻ᵛ = −∇f + Aₑᵀy − √(μ)Aᵢᵀe⁻ᵛ
  Hpˣ − Aₑᵀpʸ + Aᵢᵀdiag(e⁻²ᵛ)Aᵢpˣ + Aᵢᵀe⁻²ᵛ∘cᵢ − √(μ)Aᵢᵀe⁻ᵛ = −∇f + Aₑᵀy − √(μ)Aᵢᵀe⁻ᵛ
  Hpˣ − Aₑᵀpʸ + Aᵢᵀdiag(e⁻²ᵛ)Aᵢpˣ + Aᵢᵀe⁻²ᵛ∘cᵢ = −∇f + Aₑᵀy
  Hpˣ − Aₑᵀpʸ + Aᵢᵀdiag(e⁻²ᵛ)Aᵢpˣ = −∇f + Aₑᵀy − Aᵢᵀe⁻²ᵛ∘cᵢ
  (Hpˣ + Aᵢᵀdiag(e⁻²ᵛ)Aᵢ)pˣ − Aₑᵀpʸ = −∇f + Aₑᵀy − Aᵢᵀe⁻²ᵛ∘cᵢ
  (Hpˣ + Aᵢᵀdiag(e⁻²ᵛ)Aᵢ)pˣ − Aₑᵀpʸ = −∇f + Aₑᵀy + Aᵢᵀ(−e⁻²ᵛ∘cᵢ)
```

Substitute the new first and third rows into the system.

```
  [H + Aᵢᵀdiag(e⁻²ᵛ)Aᵢ  Aₑᵀ  0][ pˣ]    [∇f − Aₑᵀy − Aᵢᵀ(−e⁻²ᵛ∘cᵢ) ]
  [         Aₑ           0   0][−pʸ] = −[            cₑ            ]
  [         0            0   I][ pᵛ]    [1/√(μ) e⁻ᵛ∘(Aᵢpˣ + cᵢ) − e]
```

Eliminate the third row and column.

```
  [H + Aᵢᵀdiag(e⁻²ᵛ)Aᵢ  Aₑᵀ][ pˣ] = −[∇f − Aₑᵀy − Aᵢᵀ(−e⁻²ᵛ∘cᵢ)]
  [         Aₑ           0 ][−pʸ]    [           cₑ            ]
```

### Final results

In summary, the reduced 2x2 block system gives the iterates pₖˣ and pₖʸ.

```
  [H + Aᵢᵀdiag(e⁻²ᵛ)Aᵢ  Aₑᵀ][ pˣ] = −[∇f − Aₑᵀy − Aᵢᵀ(−e⁻²ᵛ∘cᵢ)]
  [         Aₑ           0 ][−pʸ]    [           cₑ            ]
```

The iterate pᵛ is given by

```
  pᵛ = 1/√(μ) e⁻ᵛ∘(Aᵢpˣ + cᵢ) − e
```

The iterates are applied like so

```
  β = 1
  αₖᵛ = 1/max(1, |pᵛ|_∞²)

  xₖ₊₁ = xₖ + αₖpₖˣ
  yₖ₊₁ = yₖ + αₖpₖʸ
  vₖ₊₁ = vₖ + αₖᵛpₖᵛ
```

where αₖ is found via backtracking line search. A filter method determines acceptance of pˣ.

Section 6 of [^3] describes how to check for local infeasibility.

## Works cited

[^1]: Nocedal, J. and Wright, S. "Numerical Optimization", 2nd. ed., Ch. 19. Springer, 2006.

[^2]: Wächter, A. and Biegler, L. "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming", 2005. [http://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf](http://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf)

[^3]: Byrd, R. and Nocedal, J. and Waltz, R. "KNITRO: An Integrated Package for Nonlinear Optimization", 2005. [https://users.iems.northwestern.edu/~nocedal/PDFfiles/integrated.pdf](https://users.iems.northwestern.edu/~nocedal/PDFfiles/integrated.pdf)

[^4]: Gu, C. and Zhu, D. "A Dwindling Filter Algorithm with a Modified Subproblem for Nonlinear Inequality Constrained Optimization", 2014. [https://sci-hub.st/10.1007/s11401-014-0826-z](https://sci-hub.st/10.1007/s11401-014-0826-z)

[^5]: Permenter, F. "Log-domain interior-point methods for convex quadratic programming", 2022. [https://arxiv.org/pdf/2212.02294](https://arxiv.org/pdf/2212.02294)
