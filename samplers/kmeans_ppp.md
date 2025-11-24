# kmeans_ppp Sampler: Diversity-Based Active Learning with K-Means++

## Overview

The **kmeans_ppp Sampler** implements a diversity-based active learning strategy that leverages a modified k-means++ algorithm to select the most informative and diverse samples from an unlabeled pool. The sampler is designed for continual learning scenarios where data arrives in sequential tasks, and the goal is to maximize model performance while minimizing labeling costs.

## Algorithm Description

### Core Principle

The kmeans_ppp sampler selects samples that maximize diversity in the feature space, ensuring that newly labeled samples provide complementary information to already labeled ones. This is achieved through a probabilistic selection mechanism based on the k-means++ initialization algorithm, adapted for active learning.

### Key Components

#### 1. **Feature Extraction**
- At each active learning cycle, the sampler extracts feature representations from all samples using the current model's feature extractor
- Features are computed via `self.agent.model.feature(x)` to obtain high-level embeddings

#### 2. **Distance-Based Diversity**
The algorithm uses Euclidean distances in the feature space to measure sample diversity:
- Samples far from already labeled data are considered more informative
- Distance calculations are weighted by optional score weights (default: uniform)

#### 3. **K-Means++ Selection Process**

##### First Active Learning Cycle (alc = 0):
- No labeled samples exist yet
- Randomly select the first sample from the unlabeled pool
- Iteratively select remaining samples based on diversity

##### Subsequent Active Learning Cycles (alc ≥ 1):
- Use previously labeled samples as **anchor points** (starting points)
- Calculate distances from unlabeled samples to these anchors
- Select new samples that are maximally diverse from both:
  - Previously labeled samples (anchors)
  - Samples selected in the current cycle

### Selection Mechanism

For each sample to be selected:

1. **Distance Calculation**: Compute squared Euclidean distances from all unlabeled samples to the nearest anchor/selected sample

2. **Probabilistic Sampling**: Sample `n_local_trials` candidate samples with probability proportional to their squared distance:
   ```
   P(sample_i) ∝ d²(sample_i, nearest_center)
   ```
   where `n_local_trials = 2 + log(n_clusters)` by default

3. **Greedy Selection**: Among the candidates, select the one that minimizes the total potential:
   ```
   best_candidate = argmin_c Σ_i min(d²(sample_i, c), current_min_distance_i)
   ```

4. **Update**: Add the selected sample to the labeled set and update distance calculations

## Mathematical Formulation

### Objective
Maximize diversity in the labeled set by selecting samples that minimize coverage redundancy:

```
S* = argmax_S Σ_{i ∈ Unlabeled} min_{j ∈ Labeled ∪ S} ||φ(x_i) - φ(x_j)||²
```

where:
- `S` is the set of samples to select in the current cycle
- `φ(x)` is the feature representation of sample `x`
- `Labeled` is the set of previously labeled samples
- `Unlabeled` is the pool of unlabeled samples

### Distance Weighting
Optional score weights can be applied to prioritize certain samples:

```
distance_weighted_i = distance²_i × score_weight_i
```

## Algorithm Pseudocode

```python
Input: 
  - X: Feature matrix for all samples
  - idx_labeled: Indices of labeled samples (empty in first cycle)
  - idx_unlabeled: Indices of unlabeled samples
  - n_samples: Number of samples to select

Output:
  - selected_indices: Indices of selected samples

1. Extract features for all samples: F = model.feature(X)

2. Initialize:
   if first_cycle:
     randomly select first sample from idx_unlabeled
     starting_points = [first_sample]
   else:
     starting_points = idx_labeled

3. Calculate initial distances:
   for each unlabeled sample u:
     dist[u] = min(||F[u] - F[s]||² for s in starting_points)

4. For i = 1 to n_samples:
   a. Sample n_local_trials candidates proportional to dist²
   b. For each candidate c:
      compute potential[c] = Σ_u min(dist[u], ||F[u] - F[c]||²)
   c. Select best = argmin(potential)
   d. Add best to selected_indices
   e. Update dist for all unlabeled samples

5. Return selected_indices
```

## Advantages

1. **Diversity Maximization**: Ensures selected samples span the feature space effectively
2. **Incremental Learning**: Leverages previously labeled samples as anchors in subsequent cycles
3. **Scalability**: Efficient distance computations using vectorized operations
4. **Probabilistic Robustness**: Multiple candidate trials reduce sensitivity to outliers
5. **Representation-Aware**: Works directly with learned feature representations

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_local_trials` | `2 + log(n_clusters)` | Number of candidate samples per selection |
| `score_weight` | `ones` | Optional weights for prioritizing samples |

## Implementation Details

### Computational Complexity
- **Per AL Cycle**: O(n_samples × n_clusters × n_local_trials × d)
  - n_samples: Number of unlabeled samples
  - n_clusters: Number of samples to select
  - d: Feature dimensionality

### Memory Requirements
- Stores distance matrix: O(n_samples)
- Feature extraction: O(total_samples × d)

## Comparison with Other Samplers

| Sampler | Selection Strategy | Considers Model | Diversity |
|---------|-------------------|-----------------|-----------|
| **RandomIter** | Random sampling | ❌ | Low |
| **TypiClust** | Clustering + Typicality | ✅ | Medium |
| **kmeans_ppp** | K-means++ Diversity | ✅ | High |

## Usage Example

```python
from samplers.kmeans_ppp import KMeansPPPSampler

# Initialize sampler
sampler = KMeansPPPSampler(
  agent=agent,
  exp_args=exp_args,
  args=args
)

# Run active learning for a task
accuracies = sampler.active_learn_task(
  run=run_id,
  task_stream=task_stream,
  task_i=task_index
)
```

## References

This implementation is adapted from:
- Arthur, D., & Vassilvitskii, S. (2007). "k-means++: The advantages of careful seeding." *SODA '07*
- Diversity-based active learning strategies for continual learning scenarios

## Notes

- The sampler is designed to work with the TSCIL-AL (Time Series Continual and Incremental Learning with Active Learning) framework
- Feature extraction uses the current model state, making selection model-aware
- The algorithm naturally balances exploration (diverse samples) with exploitation (model-informed features)
