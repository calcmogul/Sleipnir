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

Local infeasibility is only declared when the feasibility restoration phase converges to a minimizer of the constraint violation that still violates the constraints; testing for infeasibility at arbitrary iterates risks false positives. See section 3.3, p. 14 of [^2].

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
  [0    Z   0   −S ][ pˢ] = −[      Sz − μe      ]
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

Solve the fourth row for pˢ.

```
  Aᵢpˣ − pˢ = −(cᵢ − s)
  pˢ = cᵢ − s + Aᵢpˣ
```

Solve the second row for pᶻ.

```
  Σpˢ + pᶻ = μS⁻¹e − z
  pᶻ = μS⁻¹e − z − Σpˢ
```

Substitute pᶻ into the first row, expand, and cancel Aᵢᵀz from both sides.

```
  Hpˣ − Aₑᵀpʸ − Aᵢᵀ(μS⁻¹e − z − Σpˢ) = −∇f + Aₑᵀy + Aᵢᵀz
  Hpˣ − Aₑᵀpʸ − AᵢᵀμS⁻¹e + Aᵢᵀz + AᵢᵀΣpˢ = −∇f + Aₑᵀy + Aᵢᵀz
  Hpˣ − Aₑᵀpʸ − AᵢᵀμS⁻¹e + AᵢᵀΣpˢ = −∇f + Aₑᵀy
  Hpˣ − Aₑᵀpʸ + AᵢᵀΣpˢ = −∇f + Aₑᵀy + AᵢᵀμS⁻¹e
```

Substitute pˢ, distribute AᵢᵀΣ, and collect the pˣ terms.

```
  Hpˣ − Aₑᵀpʸ + AᵢᵀΣ(cᵢ − s + Aᵢpˣ) = −∇f + Aₑᵀy + AᵢᵀμS⁻¹e
  Hpˣ − Aₑᵀpʸ + AᵢᵀΣ(cᵢ − s) + AᵢᵀΣAᵢpˣ = −∇f + Aₑᵀy + AᵢᵀμS⁻¹e
  (H + AᵢᵀΣAᵢ)pˣ − Aₑᵀpʸ + AᵢᵀΣ(cᵢ − s) = −∇f + Aₑᵀy + AᵢᵀμS⁻¹e
  (H + AᵢᵀΣAᵢ)pˣ − Aₑᵀpʸ = −∇f + Aₑᵀy + Aᵢᵀ(μS⁻¹e − Σ(cᵢ − s))
  (H + AᵢᵀΣAᵢ)pˣ − Aₑᵀpʸ = −(∇f − Aₑᵀy − Aᵢᵀ(μS⁻¹e − Σ(cᵢ − s)))
```

Since pˢ and pᶻ can be recovered by back-substitution, eliminate their rows and columns.

```
  [H + AᵢᵀΣAᵢ  Aₑᵀ][ pˣ] = −[∇f − Aₑᵀy − Aᵢᵀ(μS⁻¹e − Σ(cᵢ − s))]
  [    Aₑ       0 ][−pʸ]    [                cₑ                ]
```

### Final results

In summary, the reduced 2x2 block system gives the iterates pₖˣ and pₖʸ.

```
  [H + AᵢᵀΣAᵢ  Aₑᵀ][ pˣ] = −[∇f − Aₑᵀy − Aᵢᵀ(μS⁻¹e − Σ(cᵢ − s))]
  [    Aₑ       0 ][−pʸ]    [                cₑ                ]
```

The iterates pˢ and pᶻ are given by

```
  pˢ = cᵢ − s + Aᵢpˣ
  pᶻ = μS⁻¹e − z − Σpˢ
```

The iterates are applied like so

```
  xₖ₊₁ = xₖ + αₖᵐᵃˣpₖˣ
  sₖ₊₁ = sₖ + αₖᵐᵃˣpₖˢ
  yₖ₊₁ = yₖ + αₖᵐᵃˣpₖʸ
  zₖ₊₁ = zₖ + αₖᶻpₖᶻ
```

where αₖᵐᵃˣ and αₖᶻ are computed via the fraction-to-the-boundary rule shown in equations (15a) and (15b) of [^2].

```
  αₖᵐᵃˣ = max(α ∈ (0, 1] : sₖ + αpₖˢ ≥ (1−τⱼ)sₖ)
        = max(α ∈ (0, 1] : αpₖˢ ≥ −τⱼsₖ)
  αₖᶻ = max(α ∈ (0, 1] : zₖ + αpₖᶻ ≥ (1−τⱼ)zₖ)
      = max(α ∈ (0, 1] : αpₖᶻ ≥ −τⱼzₖ)
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

## Manifold optimization

Decision variables created with `Problem::decision_variable(sphere, n)`, `Problem::decision_variable(so2)`, or `Problem::decision_variable(so3)` lie on a manifold M embedded in ℝⁿ: the unit sphere, the unit complex numbers representing SO(2), or the unit quaternions representing SO(3). The solver keeps them on the manifold instead of adding a unit-norm equality constraint. This removes a constraint and its multiplier, keeps every iterate feasible, and removes the null space a unit-norm constraint leaves in the KKT matrix. See [usage.md](usage.md#manifold-optimization) for how to use them.

### Retractions

Each iterate x is stored in the ambient space ℝⁿ, but steps p are taken in the k-dimensional tangent space of M at x. A retraction R_x(p) maps a tangent vector back onto M, with R_x(0) = x. Sleipnir uses:

```
  Sphere: R_x(p) = (x + Bp)/‖x + Bp‖
  SO(2):  R_z(p) = z exp(ip)
  SO(3):  R_q(p) = q ⊗ exp(p/2)
```

where B is an orthonormal basis for the sphere's tangent space at x, computed from a Householder reflection, and exp(φ) = [cos‖φ‖, sin‖φ‖ φ/‖φ‖] in (w, x, y, z) order. For SO(2), z = cos θ + i sin θ and p is an angle in radians. For SO(3), p is a rotation vector in the body frame. Euclidean decision variables use R_x(p) = x + p.

`Problem::decision_variable(se2)` and `Problem::decision_variable(se3)` return poses made of a Euclidean translation and an SO(2) or SO(3) rotation, so SE(2) and SE(3) use the product retraction (t + pₜ, R_R(p_R)). It differs from the group exponential, which couples the translation step to the rotation, only at second order in p, so the solver's convergence is unaffected, and it keeps the translation's derivatives independent of the rotation.

### Derivatives in the tangent space

Each iteration minimizes f(R_x(p)) over p, subject to c(R_x(p)) constraints, near p = 0. With J = ∂R/∂p at p = 0, the chain rule gives

```
  ∇ₚf = Jᵀ∇ₓf
  ∂c/∂p = AJ
  ∇²ₚₚL = Jᵀ∇²ₓₓLJ + Σᵢ (∇ₓL)ᵢ ∇²ₚₚRᵢ
```

The last term is the curvature correction. Unconstrained in ℝⁿ, ∇ₓL would be zero at a solution and the term would vanish, but on M only the tangent component Jᵀ∇ₓL is zero, and the normal component generally isn't. Dropping the term would make the Newton step use the wrong Hessian and lose quadratic convergence.

Both retractions satisfy ∇²ₚₚRᵢ = −κxᵢI, which follows from their second-order expansions

```
  Sphere: R_x(p) = x + Bp − x‖p‖²/2 + O(‖p‖³)
  SO(2):  R_z(p) = z + izp − zp²/2 + O(p³)
  SO(3):  R_q(p) = q + Jp − q‖p‖²/8 + O(‖p‖³)
```

so κ = 1 for the sphere and SO(2), and κ = 1/4 for SO(3), and the correction for each manifold block is −κ(xᵀ∇ₓL)I.

The KKT system, its inertia, and the step are k-dimensional, while the cost and constraints are evaluated at ambient points R_x(αp) during the line search. Feasibility restoration uses the same manifold, extended by the Euclidean relaxation variables.

### Constraints on manifold variables

Constraining manifold variables elementwise is ill-posed. `n == d` for unit vectors in ℝⁿ would give n equations for n − 1 degrees of freedom, so the constraint Jacobian is rank-deficient. `cos θ == cos θ*` and `sin θ == sin θ*` would give two equations for one degree of freedom. Writing a constraint on the angle instead has its own problems: sin(θ − θ*) = 0 is also satisfied at θ* + π, and a plain angle wraps at ±π.

`operator==` on `UnitVector`, `Rotation2`, `Rotation3`, `Pose2`, and `Pose3` constrains the logarithm of the relative value instead. That gives one equation per tangent dimension, with Jacobian I at the solution:

```
  Rotation2:  angle(z₂⁻¹ z₁) = 0                 (1 equation)
  Rotation3:  rotation vector of q₂⁻¹ ⊗ q₁ = 0  (3 equations)
  UnitVector: Bᵀ log_b(a) = 0                   (n − 1 equations)
  Pose2/3:    translations equal, rotations equal
```

The logarithm's only zero is the solution, and its discontinuity is at the point opposite the solution. The constraint is therefore well-conditioned from anywhere except exactly opposite. Even there, the 2D angle and 3D rotation vector pick a direction, so a heading constraint started from the opposite heading converges in one iteration. log_b(a) = atan2(‖r‖, a·b)/‖r‖ r with r = a − (a·b)b is undefined for exactly opposite unit vectors. B is an orthonormal basis of b's tangent space from a Householder reflection built around b's value when the constraint is created. If b is constant, B is exact. If b is a decision variable, B stays valid unless b moves to a single point more than 90° from that value.

Tolerance constraints don't need special handling. They're written with quantities that are smooth near the bound: the relative 2D angle, whose discontinuity is opposite the reference, and squared norms like ‖Δt‖² and ‖rotation vector‖², since the norms themselves are cone-shaped at zero, where their derivatives are undefined.

### Exponential and logarithm maps

`Rotation2(angle)` and `Rotation3(rvec)` are the exponential maps of SO(2) and SO(3), and `Twist2::exp()`/`Twist3::exp()` and `Pose2::log()`/`Pose3::log()` are the exponential and logarithm maps of SE(2) and SE(3). A twist (linear part v, angular part ω) maps to the pose with rotation exp(ω) and translation Vv, where

```
  SE(2): V = [A  −B]   A = sin θ/θ, B = (1 − cos θ)/θ
             [B   A]
  SE(3): V = I + (1 − cos θ)/θ² Ω + (θ − sin θ)/θ³ Ω²
```

with θ = ‖ω‖ and Ω the cross product matrix of ω. t ↦ exp(tξ) is the group's one-parameter subgroup generated by ξ, so exp(tξ) is the pose reached by moving with constant body-frame velocity ξ for time t. The logarithm inverts V with

```
  SE(2): V⁻¹ = [ α   θ/2]   α = (θ/2) cot(θ/2)
               [−θ/2  α ]
  SE(3): V⁻¹ = I − Φ/2 + (1 − (θ/2) cot(θ/2))/θ² Φ²
```

The coefficients are 0/0 at θ = 0, and computing them through θ = √(θ²) makes their derivatives infinite there, so they're evaluated as power series in θ² for θ² < 1 and in closed form otherwise. With 11 terms, each series' truncation error is below ε.

## Works cited

[^1]: Nocedal, J. and Wright, S. "Numerical Optimization", 2nd. ed., Ch. 19. Springer, 2006.

[^2]: Wächter, A. and Biegler, L. "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming", 2005. [http://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf](http://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf)
