# Algorithms

This page documents the algorithms Sleipnir implements.

## Reverse accumulation automatic differentiation

In reverse accumulation AD, the dependent variable to be differentiated is fixed and the derivative is computed with respect to each subexpression recursively. In a pen-and-paper calculation, the derivative of the outer functions is repeatedly substituted in the chain rule:

(∂y/∂x) = (∂y/∂w₁) ⋅ (∂w₁/∂x) = ((∂y/∂w₂) ⋅ (∂w₂/∂w₁)) ⋅ (∂w₁/∂x) = ...

In reverse accumulation, the quantity of interest is the adjoint, denoted with a bar (w̄); it is a derivative of a chosen dependent variable with respect to a subexpression w: ∂y/∂w.

Given the expression f(x₁,x₂) = sin(x₁) + x₁x₂, the computational graph is:
@mermaidinclude{reverse-autodiff}

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

The first-order necessary conditions for optimality (KKT conditions) are

```
  ∇f = 0
```

### Newton's method

Next, we'll apply Newton's method to the optimality conditions. Let H be ∂²L/∂x² and pˣ be the step for x.

```
  ∇ₓL(x + pˣ) ≈ ∇ₓL(x) + ∂²L/∂x²pˣ
  ∇ₓL(x) + Hpˣ = 0
  Hpˣ = −∇ₓL(x)
  Hpˣ = −∇f
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

The first-order necessary conditions for optimality (KKT conditions) are

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

For nonlinear models, the v update corresponds to a linear step in the dual variables, which matches the Newton linearization of z = √(μ)eᵛ

```
  pᶻ = z∘pᵛ
  zₖ₊₁ = zₖ + αₖᵛpₖᶻ = zₖ∘(e + αₖᵛpₖᵛ)
```

with its own fraction-to-the-boundary rule

```
  αₖᵛ = max(α ∈ (0, 1] : zₖ + αpₖᶻ ≥ (1−τ)zₖ)
      = min(1, τ/max(−pₖᵛ))
```

For quadratic models, v takes the additive step of [^3] with αₖᵛ = min(1, 1/|pᵛ|_∞²). The barrier parameter update keeps |pᵛ|_∞ ≤ 1 there, so the full step is usually taken. For nonlinear models, |pᵛ|_∞ can be orders of magnitude larger than 1, so that rule would stall the duals and complementarity.

The slacks are initialized to s = max(cᵢ, κ) with κ = 10⁻², and the duals are initialized with perfect complementarity (v = u, so z = μS⁻¹e). Otherwise, eᵘ⁻ᵛ and pᵛ are large for small slacks.

Local infeasibility is only declared when the feasibility restoration phase converges to a minimizer of the constraint violation that still violates the constraints; testing for infeasibility at arbitrary iterates risks false positives. See section 3.3, p. 14 of [^2].

## Log-domain interior-point method

This is a primal-dual interior-point method whose slack and dual variables are parameterized by log-domain variables u and v, as in [^3]:

```
  s = √(μ)e⁻ᵘ
  z = √(μ)eᵛ
```

This keeps s and z positive, so no fraction-to-the-boundary rule is needed for the dual variables. Using separate coordinates for s and z means the complementarity condition s∘z = μe is no longer satisfied by construction (it holds when u = v), so it's kept as an optimality condition. The barrier parameter update also follows [^3].

[^3] is for convex QPs, though, and its convergence guarantees depend on that structure. With a quadratic cost and linear constraints, x is eliminated exactly by the linear solve, so the method reduces to Newton's method on u and v alone with a simple step size rule. That doesn't hold for nonlinear programs, so the step rules follow the filter line-search method of [^2] instead. The slacks take a linear step in s rather than an additive step in u, the step size is capped by a fraction-to-the-boundary rule on the slacks, and step acceptance is determined by a filter. The resulting algorithm is closer to a standard primal-dual interior-point method than to the method of [^3], and it doesn't inherit the latter's convergence guarantees.

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

The first-order necessary conditions for optimality (KKT conditions) are

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
  s = √(μ)e⁻ᵘ
  z = √(μ)eᵛ
```
```
  ∇f − Aₑᵀy − Aᵢᵀ√(μ)eᵛ = 0
  √(μ)e⁻ᵘ∘√(μ)eᵛ − μe = 0
  cₑ = 0
  cᵢ − √(μ)e⁻ᵘ = 0

  ∇f − Aₑᵀy − √(μ)Aᵢᵀeᵛ = 0
  μeᵛ⁻ᵘ − μe = 0
  cₑ = 0
  cᵢ − √(μ)e⁻ᵘ = 0
```

### Newton's method

Next, we'll apply Newton's method to the optimality conditions. Let H be ∂²L/∂x², pˣ be the step for x, pᵘ be the step for u, pʸ be the step for y, and pᵛ be the step for v.

```
  ∇ₓL(x + pˣ, u + pᵘ, y + pʸ, v + pᵛ)
    ≈ ∇ₓL(x, u, y, v) + ∂²L/∂x²pˣ + ∂²L/∂x∂upᵘ + ∂²L/∂x∂ypʸ + ∂²L/∂x∂vpᵛ
  ∇ₓL(x, u, y, v) + Hpˣ − Aₑᵀpʸ − √(μ)Aᵢᵀeᵛ∘pᵛ = 0
  Hpˣ − Aₑᵀpʸ − √(μ)Aᵢᵀeᵛ∘pᵛ = −∇ₓL(x, u, y, v)
  Hpˣ − Aₑᵀpʸ − √(μ)Aᵢᵀeᵛ∘pᵛ = −(∇f − Aₑᵀy − √(μ)Aᵢᵀeᵛ)
```
```
  ∇ᵤL(x + pˣ, u + pᵘ, y + pʸ, v + pᵛ)
    ≈ ∇ᵤL(x, u, y, v) + ∂²L/∂u∂xpˣ + ∂²L/∂u²pᵘ + ∂²L/∂u∂ypʸ + ∂²L/∂u∂vpᵛ
  ∇ᵤL(x, u, y, v) − μeᵛ⁻ᵘ∘pᵘ + μeᵛ⁻ᵘ∘pᵛ = 0
  −μeᵛ⁻ᵘ∘pᵘ + μeᵛ⁻ᵘ∘pᵛ = −∇ᵤL(x, u, y, v)
  −μeᵛ⁻ᵘ∘pᵘ + μeᵛ⁻ᵘ∘pᵛ = −(μeᵛ⁻ᵘ − μe)
```
```
  ∇_yL(x + pˣ, u + pᵘ, y + pʸ, v + pᵛ)
    ≈ ∇_yL(x, u, y, v) + ∂²L/∂y∂xpˣ + ∂²L/∂y∂upᵘ + ∂²L/∂y²pʸ + ∂²L/∂y∂vpᵛ
  ∇_yL(x, u, y, v) + Aₑpˣ = 0
  Aₑpˣ = −∇_yL(x, u, y, v)
  Aₑpˣ = −cₑ
```
```
  ∇ᵥL(x + pˣ, u + pᵘ, y + pʸ, v + pᵛ)
    ≈ ∇ᵥL(x, u, y, v) + ∂²L/∂v∂xpˣ + ∂²L/∂v∂upᵘ + ∂²L/∂v∂ypʸ + ∂²L/∂v²pᵛ
  ∇ᵥL(x, u, y, v) + Aᵢpˣ + √(μ)e⁻ᵘ∘pᵘ = 0
  Aᵢpˣ + √(μ)e⁻ᵘ∘pᵘ = −∇ᵥL(x, u, y, v)
  Aᵢpˣ + √(μ)e⁻ᵘ∘pᵘ = −(cᵢ − √(μ)e⁻ᵘ)
```

### Matrix equation

Group them into a matrix equation.

```
  [H      0     −Aₑᵀ  −√(μ)Aᵢᵀeᵛ][pˣ]    [∇f − Aₑᵀy − √(μ)Aᵢᵀeᵛ]
  [0   −μeᵛ⁻ᵘ    0       μeᵛ⁻ᵘ  ][pᵘ] = −[     μeᵛ⁻ᵘ − μe      ]
  [Aₑ     0      0         0    ][pʸ]    [          cₑ         ]
  [Aᵢ  √(μ)e⁻ᵘ   0         0    ][pᵛ]    [    cᵢ − √(μ)e⁻ᵘ     ]
```

Invert pʸ.

```
  [H      0     Aₑᵀ  −√(μ)Aᵢᵀeᵛ][ pˣ]    [∇f − Aₑᵀy − √(μ)Aᵢᵀeᵛ]
  [0   −μeᵛ⁻ᵘ    0     μeᵛ⁻ᵘ   ][ pᵘ] = −[     μeᵛ⁻ᵘ − μe      ]
  [Aₑ     0      0       0     ][−pʸ]    [          cₑ         ]
  [Aᵢ  √(μ)e⁻ᵘ   0       0     ][ pᵛ]    [    cᵢ − √(μ)e⁻ᵘ     ]
```

Solve the second row for pᵛ.

```
  −μeᵛ⁻ᵘ∘pᵘ + μeᵛ⁻ᵘ∘pᵛ = μe − μeᵛ⁻ᵘ
  −pᵘ + pᵛ = eᵘ⁻ᵛ − e
  pᵛ = pᵘ + eᵘ⁻ᵛ − e
```

Solve the fourth row for pᵘ.

```
  Aᵢpˣ + √(μ)e⁻ᵘ∘pᵘ = −cᵢ + √(μ)e⁻ᵘ
  √(μ)e⁻ᵘ∘pᵘ = −Aᵢpˣ − cᵢ + √(μ)e⁻ᵘ
  pᵘ = e − 1/√(μ) eᵘ∘(Aᵢpˣ + cᵢ)
```

Substitute the explicit formula for pᵘ into the formula for pᵛ.

```
  pᵛ = e − 1/√(μ) eᵘ∘(Aᵢpˣ + cᵢ) + eᵘ⁻ᵛ − e
  pᵛ = eᵘ⁻ᵛ − 1/√(μ) eᵘ∘(Aᵢpˣ + cᵢ)
```

Substitute the explicit formula for pᵛ into the first row.

```
  Hpˣ − Aₑᵀpʸ − √(μ)Aᵢᵀeᵛ∘pᵛ = −∇f + Aₑᵀy + √(μ)Aᵢᵀeᵛ
  Hpˣ − Aₑᵀpʸ − √(μ)Aᵢᵀeᵛ∘(eᵘ⁻ᵛ − 1/√(μ) eᵘ∘(Aᵢpˣ + cᵢ)) = −∇f + Aₑᵀy + √(μ)Aᵢᵀeᵛ
```

Expand and simplify.

```
  Hpˣ − Aₑᵀpʸ − √(μ)Aᵢᵀeᵘ + Aᵢᵀeᵘ⁺ᵛ∘(Aᵢpˣ + cᵢ) = −∇f + Aₑᵀy + √(μ)Aᵢᵀeᵛ
  Hpˣ − Aₑᵀpʸ − √(μ)Aᵢᵀeᵘ + Aᵢᵀdiag(eᵘ⁺ᵛ)Aᵢpˣ + Aᵢᵀeᵘ⁺ᵛ∘cᵢ = −∇f + Aₑᵀy + √(μ)Aᵢᵀeᵛ
  Hpˣ − Aₑᵀpʸ + Aᵢᵀdiag(eᵘ⁺ᵛ)Aᵢpˣ = −∇f + Aₑᵀy + √(μ)Aᵢᵀeᵛ + √(μ)Aᵢᵀeᵘ − Aᵢᵀeᵘ⁺ᵛ∘cᵢ
  (H + Aᵢᵀdiag(eᵘ⁺ᵛ)Aᵢ)pˣ − Aₑᵀpʸ = −∇f + Aₑᵀy + Aᵢᵀ(√(μ)eᵘ + √(μ)eᵛ − eᵘ⁺ᵛ∘cᵢ)
```

Together with the third row, this gives the reduced system.

```
  [H + Aᵢᵀdiag(eᵘ⁺ᵛ)Aᵢ  Aₑᵀ][ pˣ] = −[∇f − Aₑᵀy − Aᵢᵀ(√(μ)eᵘ + √(μ)eᵛ − eᵘ⁺ᵛ∘cᵢ)]
  [        Aₑ            0 ][−pʸ]    [                    cₑ                    ]
```

### Final results

In summary, the reduced 2x2 block system gives the iterates pₖˣ and pₖʸ.

```
  [H + Aᵢᵀdiag(eᵘ⁺ᵛ)Aᵢ  Aₑᵀ][ pˣ] = −[∇f − Aₑᵀy − Aᵢᵀ(√(μ)eᵘ + √(μ)eᵛ − eᵘ⁺ᵛ∘cᵢ)]
  [        Aₑ            0 ][−pʸ]    [                    cₑ                    ]
```

The iterates pᵘ and pᵛ are given by

```
  pᵘ = e − 1/√(μ) eᵘ∘(Aᵢpˣ + cᵢ)
  pᵛ = pᵘ + eᵘ⁻ᵛ − e
```

When u = v (i.e., s∘z = μe), these reduce to pᵛ = pᵘ and the right-hand side Aᵢᵀ(2√(μ)eᵛ − e²ᵛ∘cᵢ) of a single log-domain coordinate.

The iterates are applied like so

```
  xₖ₊₁ = xₖ + αₖpₖˣ
  uₖ₊₁ = uₖ − ln(e − αₖpₖᵘ)
  yₖ₊₁ = yₖ + αₖpₖʸ
  vₖ₊₁ = vₖ + ln(e + αₖᵛpₖᵛ)  (nonlinear models)
  vₖ₊₁ = vₖ + αₖᵛpₖᵛ          (quadratic models)
```

The u update corresponds to a linear step in the slack variables

```
  pˢ = −s∘pᵘ
  sₖ₊₁ = sₖ + αₖpₖˢ = sₖ∘(e − αₖpₖᵘ)
```

instead of the exponential step sₖ∘exp(−αₖpₖᵘ). The linear step matches the Newton linearization, so for linear constraints cᵢ − s decreases by exactly a factor of (1 − αₖ) per step. With the exponential step, the overshoot of exp(−αₖpₖᵘ) relative to its linearization can make cᵢ − s grow.

x and u share a step size because u parameterizes the slack variables s = √(μ)e⁻ᵘ, which are part of the primal step. αₖ is found via backtracking line search starting from the fraction-to-the-boundary rule for the slack step

```
  αₖᵐᵃˣ = max(α ∈ (0, 1] : sₖ + αpₖˢ ≥ (1−τ)sₖ)
        = min(1, τ/max(pₖᵘ))
```

where τ = 0.995. Only slacks with pᵢᵘ > 0 approach the boundary, so slacks that need to grow don't limit the step. If αₖᵐᵃˣ is below the minimum step size, backtracking can't help, so feasibility restoration is invoked. A filter method determines acceptance of the step. The filter's sufficient decrease condition uses the directional derivative of the log-barrier function

```
  ϕ_μ(x, u) = f(x) − μ∑ᵢ ln(sᵢ) = f(x) + μ∑ᵢ uᵢ − μ∑ᵢ ln(√(μ))
  D_ϕ = ∇f(x)ᵀpˣ + μ∑ᵢ pᵢᵘ
```

Local infeasibility is only declared when the feasibility restoration phase converges to a minimizer of the constraint violation that still violates the constraints; testing for infeasibility at arbitrary iterates risks false positives. See section 3.3, p. 14 of [^2].

## Problem scaling

Sleipnir scales the cost and constraints to improve numerical stability. The cost and each constraint are scaled so that the largest gradient component at the starting point is at most `g_max`. For an initial point `x₀`, define

```
  d_f    = min(1, g_max / ‖∇f(x₀)‖_∞)
  d_c[j] = min(1, g_max / ‖∇c_j(x₀)‖_∞)
```

where `D_c = diag(d_c)` and `A = ∂c/∂x` is the constraint Jacobian. The scaled Lagrangian is:

```
  L(x, λ) = d_f f(x) + λᵀ D_c c(x)
```

The Hessian of the scaled Lagrangian with respect to `x` is:

```
  ∇²ₓₓL(x, λ) = d_f ∇²ₓₓf(x) + Σᵢ (D_c λ)ᵢ ∇²ₓₓcᵢ(x)
```

Recall that the original KKT matrix is:

```
  [∇²ₓₓL  Aᵀ]
  [  A    0 ]
```

Scaling the KKT matrix for this problem gives:

```
  [D_x⁻¹ ∇²ₓₓL D_x⁻¹  D_x⁻¹ Aᵀ D_c]
  [   D_c A D_x⁻¹          0      ]
```

With `D_x = I`, this reduces to:

```
  [∇²ₓₓL  Aᵀ D_c]
  [D_c A    0   ]
```

Expanding `∇²ₓₓL` yields:

```
  [d_f ∇²ₓₓf + Σᵢ (D_c λ)ᵢ ∇²ₓₓcᵢ    Aᵀ D_c]
  [            D_c A                   0   ]
```

where `D_c A` replaces `A` and `d_f ∇²ₓₓf + Σᵢ (D_c λ)ᵢ ∇²ₓₓcᵢ` replaces `∇²ₓₓf + ∇²ₓₓc` in the original KKT matrix.

The algorithm is described in more detail in section 3.8 of [^2].

## Works cited

[^1]: Nocedal, J. and Wright, S. "Numerical Optimization", 2nd. ed., Ch. 19. Springer, 2006.

[^2]: Wächter, A. and Biegler, L. "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming", 2005. [http://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf](http://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf)

[^3]: Permenter, F. "Log-domain interior-point methods for convex quadratic programming", 2022. [https://arxiv.org/pdf/2212.02294](https://arxiv.org/pdf/2212.02294)
