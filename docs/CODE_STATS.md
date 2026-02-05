# EMPO Code Statistics

*Generated: January 2, 2026*  
*Based on commit: `dc08a0d59d930f1112628a952308a440be3d7557`*

**Note:** This document reflects the codebase structure at the time of generation. Since then, files have been reorganized (e.g., `nn_based` → `learning_based`, `backward_induction.py` split into `phase1.py` and `phase2.py`). File paths in this document may not reflect current structure.

This document provides a comprehensive statistical overview of the EMPO codebase, including lines of code, complexity metrics, and defect density estimates based on established software engineering research.

## Summary

| Metric | Value |
|--------|-------|
| Total Python Files | 169 |
| Total Lines of Code (LOC) | 65,800 |
| Effective Code Lines (non-blank, non-comment) | 47,278 |
| Classes | 294 |
| Functions | 1,958 |
| Test-to-Code Ratio | 0.53 |

---

## 1. Hierarchical Lines of Code Breakdown

### 1.1 Top-Level Distribution

| Directory | Total Lines | Code Lines | Files | Classes | Functions |
|-----------|-------------|------------|-------|---------|-----------|
| `src/` | 25,456 | 19,101 | 69 | 88 | 682 |
| `tests/` | 15,185 | 10,217 | 39 | 120 | 640 |
| `examples/` | 14,297 | 10,172 | 33 | 47 | 281 |
| `vendor/` | 10,862 | 7,788 | 26 | 39 | 355 |
| **Total** | **65,800** | **47,278** | **167** | **294** | **1,958** |

### 1.2 Source Code (`src/`) Breakdown

#### Main Packages

| Package | Files | Total Lines | Code Lines | Classes | Functions |
|---------|-------|-------------|------------|---------|-----------|
| `src/empo/` | 65 | 24,507 | 18,399 | 84 | 667 |
| `src/envs/` | 2 | 484 | 352 | 4 | 7 |
| `src/llm_hierarchical_modeler/` | 2 | 465 | 350 | 0 | 8 |

#### `src/empo/` Subpackage Hierarchy

```
src/empo/                                    7 files    5,852 lines   (4,354 code)   25 cls  160 fn
├── hierarchical/                            1 file        26 lines      (18 code)    0 cls    0 fn
├── nn_based/                                9 files      863 lines     (662 code)    8 cls   39 fn
│   ├── multigrid/                          11 files    4,515 lines   (3,398 code)    8 cls   91 fn
│   │   └── phase2/                          8 files    3,113 lines   (2,475 code)    8 cls   50 fn
│   ├── phase2/                             14 files    5,975 lines   (4,320 code)   19 cls  173 fn
│   │   └── lookup/                          7 files    1,805 lines   (1,433 code)    8 cls  113 fn
│   └── transport/                           8 files    2,358 lines   (1,739 code)    8 cls   41 fn
└── TOTAL:                                  65 files   24,507 lines  (18,399 code)   84 cls  667 fn
```

#### Root `src/empo/` Files Detail

| File | Lines | Code Lines |
|------|-------|------------|
| `world_model.py` | 823 | 574 |
| `backward_induction.py` | 1,136 | 882 |
| `human_policy_prior.py` | 1,195 | 864 |
| `multigrid.py` | 931 | 696 |
| `possible_goal.py` | 455 | 352 |
| `transport.py` | 1,219 | 916 |
| `__init__.py` | 88 | 70 |

### 1.3 Vendor Code Breakdown

#### `vendor/multigrid/`

| File | Lines | Code Lines |
|------|-------|------------|
| `gym_multigrid/multigrid.py` | 5,011 | 3,601 |
| `gym_multigrid/rendering.py` | 163 | 115 |
| `gym_multigrid/window.py` | 90 | 54 |
| `gym_multigrid/envs/collect_game.py` | 98 | 75 |
| `gym_multigrid/envs/soccer_game.py` | 118 | 97 |
| **Subtotal** | **5,532** | **3,984** |

#### `vendor/ai_transport/`

| Path | Lines | Code Lines |
|------|-------|------------|
| `ai_transport/envs/transport_env.py` | 1,749 | 1,245 |
| `ai_transport/envs/clustering.py` | 341 | 267 |
| `ai_transport/policies/human_policies.py` | 299 | 230 |
| `ai_transport/policies/vehicle_policies.py` | 325 | 238 |
| `examples/` (8 files) | 1,409 | 1,059 |
| `tests/test_transport_env.py` | 982 | 611 |
| **Subtotal** | **5,304** | **3,804** |

---

## 2. Largest Files

| Rank | File | Lines |
|------|------|-------|
| 1 | `vendor/multigrid/gym_multigrid/multigrid.py` | 5,012 |
| 2 | `src/empo/nn_based/phase2/trainer.py` | 2,960 |
| 3 | `vendor/ai_transport/ai_transport/envs/transport_env.py` | 1,750 |
| 4 | `examples/phase2/phase2_robot_policy_demo.py` | 1,414 |
| 5 | `src/empo/transport.py` | 1,220 |
| 6 | `src/empo/human_policy_prior.py` | 1,195 |
| 7 | `src/empo/backward_induction.py` | 1,137 |
| 8 | `src/empo/nn_based/multigrid/state_encoder.py` | 1,039 |
| 9 | `tests/test_transport_env.py` | 998 |
| 10 | `src/empo/nn_based/multigrid/phase2/trainer.py` | 997 |

---

## 3. Complexity Metrics

### 3.1 Cyclomatic Complexity by Directory

| Directory | Functions | Decision Points | Avg CC |
|-----------|-----------|-----------------|--------|
| `src/` | 682 | 1,828 | 3.68 |
| `tests/` | 640 | 541 | 1.85 |
| `examples/` | 281 | 1,128 | 5.01 |
| `vendor/` | 355 | 1,286 | 4.62 |

*Note: Cyclomatic Complexity (CC) = 1 + decision points per function. CC < 10 is generally considered acceptable.*

### 3.2 Highest Complexity Files

Files with avg cyclomatic complexity ≥ 8.0 (with at least 5 functions):

| Avg CC | Functions | File |
|--------|-----------|------|
| 12.88 | 8 | `src/empo/nn_based/multigrid/neural_policy_prior.py` |
| 12.00 | 8 | `src/empo/nn_based/multigrid/phase2/trainer.py` |
| 10.25 | 8 | `examples/diagnostics/bellman_backward_induction.py` |
| 9.00 | 8 | `examples/transport/transport_stress_test_demo.py` |
| 8.80 | 5 | `examples/multigrid/heuristic_key_door_demo.py` |
| 8.75 | 16 | `src/empo/backward_induction.py` |
| 8.60 | 5 | `src/empo/nn_based/transport/feature_extraction.py` |
| 8.22 | 9 | `examples/diagnostics/dag_and_episode_example.py` |
| 8.17 | 6 | `examples/transport/transport_two_cluster_demo.py` |
| 8.00 | 16 | `src/empo/nn_based/multigrid/path_distance.py` |

### 3.3 Function Length Distribution

| Metric | Value |
|--------|-------|
| Total Functions | 1,958 |
| Average Length | 27.3 lines |
| Median Length | 15 lines |
| Maximum Length | 559 lines |
| Functions > 100 lines | 78 (4.0%) |
| Functions > 50 lines | 274 (14.0%) |
| Functions ≤ 10 lines | 712 (36.4%) |

---

## 4. Code Quality Indicators

### 4.1 Line Composition

| Directory | Total | Code | Blank | Comments | Comment % |
|-----------|-------|------|-------|----------|-----------|
| `src/` | 25,456 | 19,101 | 4,327 | 2,028 | 8.0% |
| `tests/` | 15,185 | 10,217 | 3,529 | 1,439 | 9.5% |
| `examples/` | 14,297 | 10,172 | 2,623 | 1,502 | 10.5% |
| `vendor/` | 10,862 | 7,788 | 1,942 | 1,132 | 10.4% |

### 4.2 Code Ratios

| Ratio | Value | Assessment |
|-------|-------|------------|
| Test-to-Code | 0.53 | Good (target ≥ 0.5) |
| Example-to-Code | 0.53 | Excellent documentation |
| Comment Density | 9.4% | Acceptable |

### 4.3 Code Health Markers

| Metric | Count |
|--------|-------|
| TODO/FIXME/XXX/HACK markers | 4 |
| Import statements | 1,236 |
| YAML config files | 29 (608 lines) |
| Markdown documentation | 7,114 lines |

---

## 5. Defect Density Estimates

### 5.1 Bug Estimation Models

Software engineering research provides several models for estimating defect density. We apply these to the EMPO codebase:

#### Model 1: Industry Average (Capers Jones)

Based on Capers Jones' research, typical defect densities range from:
- **Best-in-class**: 0.5 defects per 1,000 LOC
- **Industry average**: 1-25 defects per 1,000 LOC  
- **Research/scientific code**: 15-50 defects per 1,000 LOC (often less tested)

| Assumption | Own Code (39,490 LOC) | With Vendor (47,278 LOC) |
|------------|----------------------|--------------------------|
| Best-in-class (0.5/KLOC) | ~20 bugs | ~24 bugs |
| Good quality (3/KLOC) | ~118 bugs | ~142 bugs |
| Industry avg (15/KLOC) | ~592 bugs | ~709 bugs |

#### Model 2: Code Complete (Steve McConnell)

McConnell's data suggests:
- **Tested code**: 1-3 defects per 1,000 LOC after unit testing
- **Formally verified**: 0.1 defects per 1,000 LOC

Given EMPO's test-to-code ratio of 0.53:

| Component | LOC | Estimated Bugs (2/KLOC) |
|-----------|-----|------------------------|
| Source code | 19,101 | ~38 |
| Examples | 10,172 | ~20 |
| Vendor | 7,788 | ~16 |
| **Total** | **37,061** | **~74** |

#### Model 3: Halstead Complexity-Based

Using Halstead's model where defects scale with complexity:
- Higher CC files tend to have more defects
- Files with CC > 10 have ~2-4x higher defect rates

Based on our complexity analysis:
- High complexity files (CC > 8): ~10 files × 3 bugs = ~30 bugs
- Medium complexity (CC 4-8): ~50 files × 1 bug = ~50 bugs  
- Low complexity (CC < 4): ~90 files × 0.3 bugs = ~27 bugs
- **Estimated total: ~107 bugs**

### 5.2 Risk Areas (High Defect Probability)

Based on size and complexity, these areas warrant extra attention:

| Priority | File/Module | Risk Factors |
|----------|-------------|--------------|
| 🔴 High | `nn_based/phase2/trainer.py` (2,960 lines) | Largest source file, complex state management |
| 🔴 High | `nn_based/multigrid/neural_policy_prior.py` | Highest avg CC (12.88) |
| 🔴 High | `backward_induction.py` | High CC (8.75), 3 TODO markers |
| 🟡 Medium | `multigrid/phase2/trainer.py` | High CC (12.00) |
| 🟡 Medium | `transport.py` (1,220 lines) | Large file, domain complexity |
| 🟡 Medium | `human_policy_prior.py` (1,195 lines) | Core algorithm, complex logic |

### 5.3 Consolidated Bug Estimate

Averaging across models and accounting for:
- Strong test coverage (0.53 ratio)
- Active development (few TODO markers)
- Complex algorithmic code (research software)

**Estimated defect range: 50-150 latent bugs** in the own codebase (excluding vendor).

Most likely:
- **~20-30 critical/blocking bugs**
- **~30-50 moderate bugs**
- **~30-70 minor/cosmetic issues**

---

## 6. Comparison with Industry Benchmarks

| Metric | EMPO | Industry Typical | Assessment |
|--------|------|------------------|------------|
| Avg function length | 27.3 lines | 20-30 lines | ✅ Normal |
| Avg cyclomatic complexity | 3.68 | 4-6 | ✅ Good |
| Test-to-code ratio | 0.53 | 0.3-0.5 | ✅ Good |
| Comment density | 9.4% | 10-20% | ⚠️ Slightly low |
| TODO markers | 4 | Varies | ✅ Low technical debt |
| Max function length | 559 lines | < 100 recommended | ⚠️ Some refactoring needed |
| Files > 1000 lines | 10 | Minimize | ⚠️ Consider splitting |

---

## 7. Recommendations

### High Priority
1. **Refactor high-complexity files** - Target files with CC > 10, especially `neural_policy_prior.py` and trainers
2. **Split large functions** - 78 functions exceed 100 lines; aim for < 50 lines each
3. **Address TODO markers** - Resolve 3 TODOs in `backward_induction.py`

### Medium Priority
4. **Increase comment density** - Especially in `src/empo/nn_based/` which contains core algorithms
5. **Consider splitting `multigrid.py`** (5,011 lines) - Even as vendor code, modifications may introduce bugs
6. **Add integration tests** for high-risk modules

### Ongoing
7. **Monitor complexity trends** as features are added
8. **Maintain test-to-code ratio** above 0.5

---

## References

1. Jones, C. (2008). *Applied Software Measurement*. McGraw-Hill.
2. McConnell, S. (2004). *Code Complete, 2nd Edition*. Microsoft Press.
3. Halstead, M.H. (1977). *Elements of Software Science*. Elsevier.
4. Lipow, M. (1982). "Number of Faults per Line of Code." *IEEE Transactions on Software Engineering*.
5. Basili, V.R. & Perricone, B.T. (1984). "Software Errors and Complexity." *Communications of the ACM*.
