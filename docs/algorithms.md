# Algorithms

This page documents the algorithms Sleipnir implements.

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

Local infeasibility is only declared when the feasibility restoration phase converges to a minimizer of the constraint violation that still violates the constraints; testing for infeasibility at arbitrary iterates risks false positives. See section 3.3, p. 14 of [^2].

## Log-domain interior-point method

This is a primal-dual interior-point method whose slack and dual variables are parameterized by log-domain variables v, as in [^5]:

```
  s = √(μ)e⁻ᵛ
  z = √(μ)eᵛ
```

This keeps s and z positive and satisfies the complementarity condition s∘z = μe exactly, so z is tied to the slacks via z = μS⁻¹e and no fraction-to-the-boundary rule is needed for the dual variables. To keep the duals bounded, the constraints are relaxed by μw and reduced at the same rate as the barrier parameter, as in the one-phase method of [^4]. The barrier parameter update follows [^4] as well; see [Barrier parameter update](#barrier-parameter-update).

[^5] is for convex QPs, though, and its convergence guarantees depend on that structure. With a quadratic cost and linear constraints, x is eliminated exactly by the linear solve, so the method reduces to Newton's method on v alone with a simple step size rule. That doesn't hold for nonlinear programs, so the step rules follow the filter line-search method of [^2] instead. The slacks take a linear step in s rather than an additive step in v, the step size is capped by a fraction-to-the-boundary rule on the slacks, and step acceptance is determined by a filter. The resulting algorithm is closer to a standard primal-dual interior-point method than to the method of [^5], and it doesn't inherit the latter's convergence guarantees.

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
  s = √(μ)e⁻ᵛ
  z = √(μ)eᵛ
```
```
  ∇f − Aₑᵀy − Aᵢᵀ√(μ)eᵛ = 0
  cₑ = 0
  cᵢ − √(μ)e⁻ᵛ = 0

  ∇f − Aₑᵀy − √(μ)Aᵢᵀeᵛ = 0
  cₑ = 0
  cᵢ − √(μ)e⁻ᵛ = 0
```

The complementarity condition is now always satisfied, so it can be omitted.

### Newton's method

Next, we'll apply Newton's method to the optimality conditions. Let H be ∂²L/∂x², pˣ be the step for x, pʸ be the step for y, and pᵛ be the step for v.

```
  ∇ₓL(x + pˣ, y + pʸ, v + pᵛ)
    ≈ ∇ₓL(x, y, v) + ∂²L/∂x²pˣ + ∂²L/∂x∂ypʸ + ∂²L/∂x∂vpᵛ
  ∇ₓL(x, y, v) + Hpˣ − Aₑᵀpʸ − √(μ)Aᵢᵀeᵛ∘pᵛ = 0
  Hpˣ − Aₑᵀpʸ − √(μ)Aᵢᵀeᵛ∘pᵛ = −∇ₓL(x, y, v)
  Hpˣ − Aₑᵀpʸ − √(μ)Aᵢᵀeᵛ∘pᵛ = −(∇f − Aₑᵀy − √(μ)Aᵢᵀeᵛ)
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
  ∇ᵥL(x, y, v) + Aᵢpˣ + √(μ)e⁻ᵛ∘pᵛ = 0
  Aᵢpˣ + √(μ)e⁻ᵛ∘pᵛ = −∇ᵥL(x, y, v)
  Aᵢpˣ + √(μ)e⁻ᵛ∘pᵛ = −(cᵢ − √(μ)e⁻ᵛ)
```

### Matrix equation

Group them into a matrix equation.

```
  [H   −√(μ)Aᵢᵀeᵛ][pˣ] = −[∇f − √(μ)Aᵢᵀeᵛ − μβ₁e]
  [Aᵢ   √(μ)e⁻ᵛ  ][pᵛ]    [  cᵢ − √(μ)e⁻ᵛ + μw  ]
```

Solve the second row for pᵛ.

```
  Aᵢpˣ + √(μ)e⁻ᵛ∘pᵛ = −cᵢ + √(μ)e⁻ᵛ − μw
  √(μ)e⁻ᵛ∘pᵛ = −Aᵢpˣ − cᵢ + √(μ)e⁻ᵛ − μw
  pᵛ = −1/√(μ) Aᵢeᵛ∘pˣ − 1/√(μ) eᵛ∘cᵢ + e − √(μ)eᵛ∘w
  pᵛ = e − 1/√(μ) eᵛ∘(Aᵢpˣ + cᵢ) − √(μ)eᵛ∘w
```

Substitute the explicit formula for pᵛ into the first row.

```
  Hpˣ − √(μ)Aᵢᵀeᵛ∘pᵛ = −∇f + √(μ)Aᵢᵀeᵛ + μβ₁e
  Hpˣ − √(μ)Aᵢᵀeᵛ∘(e − 1/√(μ) eᵛ∘(Aᵢpˣ + cᵢ) − √(μ)eᵛ∘w) = −∇f + √(μ)Aᵢᵀeᵛ + μβ₁e
```

Expand and simplify.

```
  Hpˣ − Aᵢᵀeᵛ∘(√(μ) − eᵛ∘(Aᵢpˣ + cᵢ) − eᵛ∘μw) = −∇f + √(μ)Aᵢᵀeᵛ + μβ₁e
  Hpˣ − √(μ)Aᵢᵀeᵛ + Aᵢᵀe²ᵛ∘(Aᵢpˣ + cᵢ) + Aᵢᵀe²ᵛ∘μw = −∇f + √(μ)Aᵢᵀeᵛ + μβ₁e
  Hpˣ − √(μ)Aᵢᵀeᵛ + Aᵢᵀdiag(e²ᵛ)Aᵢpˣ + Aᵢᵀe²ᵛ∘(cᵢ + μw) = −∇f + √(μ)Aᵢᵀeᵛ + μβ₁e
  Hpˣ + Aᵢᵀdiag(e²ᵛ)Aᵢpˣ + Aᵢᵀe²ᵛ∘(cᵢ + μw) = −∇f + 2√(μ)Aᵢᵀeᵛ + μβ₁e
  Hpˣ + Aᵢᵀdiag(e²ᵛ)Aᵢpˣ = −∇f + 2√(μ)Aᵢᵀeᵛ − Aᵢᵀe²ᵛ∘(cᵢ + μw) + μβ₁e
  (Hpˣ + Aᵢᵀdiag(e²ᵛ)Aᵢ)pˣ = −∇f + 2√(μ)Aᵢᵀeᵛ − Aᵢᵀe²ᵛ∘(cᵢ + μw) + μβ₁e
  (Hpˣ + Aᵢᵀdiag(e²ᵛ)Aᵢ)pˣ = −∇f + Aᵢᵀ(2√(μ)eᵛ − e²ᵛ∘(cᵢ + μw)) + μβ₁e
```

Substitute the new first and second rows into the system.

```
  [H + Aᵢᵀdiag(e²ᵛ)Aᵢ  0][pˣ] = −[∇f − Aᵢᵀ(2√(μ)eᵛ − e²ᵛ∘(cᵢ + μw)) − μβ₁e]
  [        0           I][pᵛ]    [  e − 1/√(μ) eᵛ∘(Aᵢpˣ + cᵢ) − √(μ)eᵛ∘w  ]
```

Eliminate the second row and column.

```
  [H + Aᵢᵀdiag(e²ᵛ)Aᵢ][pˣ] = −[∇f − Aᵢᵀ(2√(μ)eᵛ − e²ᵛ∘(cᵢ + μw)) − μβ₁e]
```

### Final results

In summary, the following system gives the iterate pₖˣ.

```
  [H + Aᵢᵀdiag(e²ᵛ)Aᵢ][pˣ] = −[∇f − Aᵢᵀ(2√(μ)eᵛ − e²ᵛ∘(cᵢ + μw)) − μβ₁e]
```

The iterate pᵛ is given by

```
  pᵛ = e − 1/√(μ) eᵛ∘(Aᵢpˣ + cᵢ) − √(μ)eᵛ∘w
```

The iterates are applied like so

```
  xₖ₊₁ = xₖ + αₖpₖˣ
  yₖ₊₁ = yₖ + αₖpₖʸ
  vₖ₊₁ = vₖ − ln(e − αₖpₖᵛ)
```

The v update corresponds to a linear step in the slack variables

```
  pˢ = −s∘pᵛ
  sₖ₊₁ = sₖ + αₖpₖˢ = sₖ∘(e − αₖpₖᵛ)
```

instead of the exponential step sₖ∘exp(−αₖpₖᵛ). The linear step matches the Newton linearization, so for linear constraints cᵢ − s decreases by exactly a factor of (1 − αₖ) per step. With the exponential step, the overshoot of exp(−αₖpₖᵛ) relative to its linearization can make cᵢ − s grow. The dual variables follow from z = μS⁻¹e.

x and v share a step size because v parameterizes the slack variables s = √(μ)e⁻ᵛ, which are part of the primal step. αₖ is found via backtracking line search starting from the fraction-to-the-boundary rule for the slack step

```
  αₖᵐᵃˣ = max(α ∈ (0, 1] : sₖ + αpₖˢ ≥ (1−τ)sₖ)
        = min(1, τ/max(pₖᵛ))
```

where τ = 0.995. Only slacks with pᵢᵛ > 0 approach the boundary, so slacks that need to grow don't limit the step. If αₖᵐᵃˣ is below the minimum step size, backtracking can't help, so feasibility restoration is invoked. A filter method determines acceptance of the step. The filter's sufficient decrease condition uses the directional derivative of the log-barrier function

```
  ϕ_μ(x, v) = f(x) − μ∑ᵢ ln(sᵢ) = f(x) + μ∑ᵢ vᵢ − μ∑ᵢ ln(√(μ))
  D_ϕ = ∇f(x)ᵀpˣ + μ∑ᵢ pᵢᵛ
```

Local infeasibility is only declared when the feasibility restoration phase converges to a minimizer of the constraint violation that still violates the constraints; testing for infeasibility at arbitrary iterates risks false positives. See section 3.3, p. 14 of [^2].

The step targets the relaxed constraint cᵢ − s + μw = 0 (w = e), so the filter measures constraint violation as ‖cᵢ − s + μw‖₁, and feasibility restoration relaxes cᵢ(x) + μw − p + n ≥ 0 instead of cᵢ(x) − p + n ≥ 0. Otherwise, the violation measure can't fall below μ‖w‖₁ near the barrier subproblem's solution, and the filter rejects productive steps.

### Barrier parameter update

The barrier parameter is updated when the barrier subproblem is approximately solved (its KKT error, including the μw and μβ₁e perturbations, is at most 10μ) and the previous step satisfied |pᵛ|_∞ ≤ 1.

Lowering μ on its own doesn't work with the relaxed constraints. At the barrier subproblem's solution, an active constraint sits at cᵢ ≈ s − μw, so lowering μ requires each active slack to absorb the drop in μw in one step. That fails when sᵢ ≪ μ. Instead, μ is reduced along with the relaxed infeasibility by an aggressive step as in [^4]. Linearize the perturbed KKT conditions in x, v, and μ with dμ = −ημ for η ∈ (0, 1], so the barrier parameter after a step of size α is

```
  μ⁺ = (1 − αη)μ
```

Substituting z = √(μ)eᵛ and s = √(μ)e⁻ᵛ, the derivatives with respect to μ are z/(2μ) and −s/(2μ). Eliminating pᵛ as before gives a system with the same matrix as the normal step.

```
  [H + Aᵢᵀdiag(e²ᵛ)Aᵢ][pˣ] = −[∇f − Aᵢᵀ((2 − η)√(μ)eᵛ − e²ᵛ∘(cᵢ + (1 − η)μw)) − (1 − η)μβ₁e]
```

The slacks take the linear step sₖ₊₁ = sₖ∘(e − αqᵛ) where

```
  qᵛ = e − 1/√(μ) eᵛ∘(Aᵢpˣ + cᵢ + (1 − η)μw)
```

Since sₖ₊₁ = √(μ⁺)exp(−vₖ₊₁),

```
  vₖ₊₁ = vₖ − ln(e − αqᵛ) + ½ln(1 − αη)
```

η = 0 recovers the normal step. For linear constraints, cᵢ − s + μw shrinks by exactly (1 − αη) along with μ, so the relaxed infeasibility and μ go to zero at the same rate, which keeps the duals bounded.

A pure affine step (η = 1) is quickly blocked by the fraction-to-the-boundary rule, so η is chosen with Mehrotra's heuristic

```
  η = 1 − (1 − α_aff)³
```

where α_aff is the fraction-to-the-boundary step size of the η = 1 direction. α starts at the fraction-to-the-boundary step size of qᵛ, capped so μ⁺ ≥ μₘᵢₙ, and is halved until the barrier subproblem's KKT error at the trial iterate is at most 10μ⁺. The filter is reset after a successful step.

If no step reduces μ by at least 1% (αη ≥ 0.01), the central path is blocked by the boundary (e.g., the barrier subproblem's solution is on a branch that becomes infeasible as μ → 0). Repeated short steps would converge to a μ > 0 instead, so μ is reduced to 0.2μ with the slacks held fixed (v shifted by ln(√(μ⁺)/√(μ))). The iterate then violates the tighter relaxation, which the normal step and feasibility restoration can reduce.

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

[^3]: Gu, C. and Zhu, D. "A Dwindling Filter Algorithm with a Modified Subproblem for Nonlinear Inequality Constrained Optimization", 2014. [https://sci-hub.st/10.1007/s11401-014-0826-z](https://sci-hub.st/10.1007/s11401-014-0826-z)

[^4]: Hinder, O. and Ye, Y. "A one-phase interior point method for nonconvex optimization", 2018. [https://arxiv.org/pdf/1801.03072.pdf](https://arxiv.org/pdf/1801.03072.pdf)

[^5]: Permenter, F. "Log-domain interior-point methods for convex quadratic programming", 2022. [https://arxiv.org/pdf/2212.02294](https://arxiv.org/pdf/2212.02294)
