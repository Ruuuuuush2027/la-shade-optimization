# MILP Analysis & Pareto Frontier Visualization

## Question 1: Would MILP Have Worked Here?

### **Short Answer: Yes, but with important caveats**

### MILP Formulation

Your implementation exists in `methods/milp_solver.py`:

```python
maximize: Σ(reward_i × x_i) for all i
subject to:
    Σ x_i = k                    (select exactly k shades)
    x_i ∈ {0, 1}                 (binary decision)
    x_i + x_j ≤ 1  ∀ij where dist(i,j) < 500m  (spacing constraint)
```

### Advantages of MILP

✅ **Optimal Solution (with caveat)**
- Guarantees global optimum for the *linearized* problem
- No greedy approximation error
- Theoretical optimality gap reporting

✅ **Handles Hard Constraints Elegantly**
- Spacing constraints (min 500m) are natural in MILP
- Budget constraints (exactly k locations)
- Accessibility requirements
- Olympic venue coverage minimums

✅ **Fast for Small k**
- k=10: Should solve in seconds to minutes
- k=100: Might take 10-60 minutes (depends on n)
- Modern solvers (Gurobi, CPLEX, CBC) are highly optimized

✅ **Interpretable**
- Clear mathematical formulation
- Dual variables show marginal value of constraints
- Sensitivity analysis available

### Critical Limitations

❌ **Non-Linear Reward Functions Are Problematic**

Your Approach 2 (Hierarchical/Multiplicative) is **NOT linear**:
```python
# This is non-linear!
reward = base_score × heat_equity_multiplier × olympic_multiplier
```

MILP requires linearization:
- Approximation needed
- Loses optimality guarantee
- May perform worse than greedy for non-linear objectives

❌ **State-Dependent Rewards**

Your reward functions have **diminishing returns** and **marginal dependencies**:
```python
# Reward of location i depends on what's already selected
reward(i | selected) ≠ reward(i)
```

This violates MILP's assumption of **independent objective coefficients**.

**Your code's workaround** (line 59-60):
```python
# Use empty state for approximation
rewards[idx] = reward_function.calculate_reward([], idx)
```

This is an **approximation** that ignores:
- Diminishing marginal utility
- Spatial saturation effects
- Compounding effects in Approach 2

❌ **Scalability Concerns**

For n candidate locations:
- Variables: O(n) binary variables
- Constraints: O(n²) distance constraints (if many locations within 500m)
- For n=1000, k=100: ~500,000 distance constraints possible

Modern solvers can handle this, but runtime grows significantly.

### When MILP Would Excel

**MILP is ideal for:**

1. **Approach 1 with linear approximation**
   ```python
   # Simple weighted sum (can be linearized)
   reward = 0.30·heat + 0.25·pop + 0.18·equity + ...
   ```

2. **Hard constraints that greedy struggles with**
   - "At least 3 shades in Inglewood"
   - "No more than 30% budget in Downtown"
   - "Cover at least 80% of Olympic venues"

3. **Small to medium k (k ≤ 50)**
   - Optimal solutions in reasonable time

4. **Multi-period planning**
   ```
   Year 1: Place k₁ locations
   Year 2: Place k₂ locations (given Year 1)
   Total budget constraint across years
   ```

### When Greedy Is Better

**Greedy excels when:**

1. **Objective is submodular** (diminishing returns)
   - Guarantees (1-1/e) ≈ 63% approximation
   - Your problem likely has submodular structure!

2. **Non-linear, state-dependent rewards**
   - Approach 2's multiplicative structure
   - Saturation effects
   - Marginal utility calculations

3. **Large k (k ≥ 100)**
   - Greedy: O(k·n) - very fast
   - MILP: Exponential worst case

4. **Need quick iterations during development**

### Comparison Table

| Aspect | MILP | Greedy |
|--------|------|--------|
| **Optimality** | Optimal for linear | (1-1/e) for submodular |
| **Speed (k=10)** | Seconds to minutes | Seconds |
| **Speed (k=100)** | Minutes to hours | Seconds |
| **Non-linear objectives** | Requires approximation ❌ | Handles naturally ✅ |
| **Hard constraints** | Excellent ✅ | Difficult ❌ |
| **State-dependent rewards** | Approximation needed ❌ | Native support ✅ |
| **Interpretability** | High (duals, gaps) | Medium |
| **Implementation complexity** | Medium (needs solver) | Low |

---

## Why MILP Wasn't Tested in Your Results

Looking at your results directory, MILP outputs are missing. Likely reasons:

1. **PuLP not installed** → Falls back to greedy (line 37-41)
2. **State-dependent rewards** → Implementation knows MILP approximation is poor
3. **Time constraints** → Greedy was "good enough"
4. **Non-linear Approach 2** → MILP not suitable

---

## Recommendation: When to Use MILP

### Use MILP for:

**Scenario 1: Approach 1 + Hard Constraints**
```python
maximize: Σ(reward_i × x_i)
subject to:
    Σ x_i = k
    Spacing constraints
    At least 2 shades within 500m of Olympic venues  ← Hard constraint
    At least 40% of shades in high-SOVI areas       ← Hard constraint
```

**Scenario 2: Multi-Objective with Weighted Sum**
```python
maximize: w₁·Σheat_i·x_i + w₂·Σpop_i·x_i + w₃·Σsovi_i·x_i
subject to: constraints...
```

Then sweep over weights w₁, w₂, w₃ to approximate Pareto frontier.

### Use Greedy for:

- Approach 2 (Multiplicative/Hierarchical)
- Quick iterations and development
- Large k (k ≥ 100)
- When MILP runtime is prohibitive

### Hybrid Approach (Best of Both):

1. **Use Greedy** for initial solution (fast)
2. **Use MILP** with greedy solution as warm start
3. Add hard constraints to MILP that greedy can't handle
4. Compare solutions

---

## Question 2: Pareto Front & Rosetta Plot

### What is a Pareto Front?

**Definition**: Set of solutions where improving one objective requires worsening another.

For your problem:
- **Objective 1**: Maximize heat mitigation
- **Objective 2**: Maximize socio-vulnerability coverage
- **Objective 3**: Maximize population served
- **Objective 4**: Maximize Olympic coverage
- **Objective 5**: Minimize inequity (Gini)

**Pareto frontier** = All non-dominated solutions

### Current Data: 2D Pareto Fronts

You don't have full Pareto optimization results yet, but we can:

1. **Plot existing solutions** (Greedy, RL, Random, etc.) in objective space
2. **Identify dominated solutions**
3. **Show trade-off structure**

### What is a Rosetta Plot?

**Rosetta plot** (aka parallel coordinates plot):
- X-axis: Different objectives
- Y-axis: Normalized performance (0-1)
- Each line = One solution
- Shows high-dimensional trade-offs at a glance

Perfect for comparing:
- Approach 1 vs Approach 2
- Different methods (Greedy vs RL)
- Multi-objective solutions

---

## Visualization Code

I'll create both visualizations below.
