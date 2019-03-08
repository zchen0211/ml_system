# XGBoost: A Scalable Tree Boosting System
 # KDD 2016
  Regularized Learning Objective
  Gradient Tree Boosting
  Shrinkage and Column Subsampling (Reduce overfitting)
   times a scalar
   sub-sample
 Split Finding Algorithms
  Basic Exact Greedy Algorithm (for each sorted index, compute score)
  Approximate Algorithm (Percentile)
  Weighted Quantile Sketch
  Sparsity-Aware Split Finding
 System Design
  Most time-consuming: sorting
  Cache Aware Access
  Blocks for Out-of-core Computation
