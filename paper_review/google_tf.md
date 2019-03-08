# Tensorflow

## Papers
- Tensorflow: Large-scale machine learning on heterogeneous distributed systems, 2016
- TensorFlow: A System for Large-Scale Machine Learning, OSDI 2016

## DAG Graph Optimization
 Node Placement to decide device: first run a simulation; greedy selection based on completion time;
 Common subgraph elimination [8];
 Scheduling: node execution as late as possible, reduce peak memory consumption;
 Lossy compression: robust, IEEE 32-bit float v.s. 16-bit in mantissa
 Back-propagation: BFS graph; symbol-to-number (Caffe, Torch); symbol-to-symbol (TF) considerable performance cost, increased memory overhead (TF, Theano)
 Control-flow: if and while, unrolled operation for RNN
 To train large models (NLP): sparse embedding layers; Gather and Stitch; sampled softmax;

## Dynamic DAG Graph Optimization
 Widely used in parse-trees, logical terms and molecular graphs; 
 Tensorflow Fold:
   Different graphs -> a whole graph;
   Apply while loop: each iteration compute different depth; (c) Apply gather for input and concat for output;

## Data-Flow
 Elements: Tensors, Operations, Stateful operations (Variable, queues)
 Data flow partial and concurrent execution: feed, fetch, step
 Directed, acyclic; R_op to solve for Jacobi matrix
 Scheduling: ASAP (as-soon-as-possible), ALAP (as-late-as-possible), list-based schedul- ing (general ASAP + priority-based ready list); Longest path through data flow determines minimum schedule length
 Scan to sidestep cycles;
 Graph Optimization: canonicalize, stabilize, specialize, GPU, in-place, scan;

 Distributed data-flow [24], timely data-flow (Naiad) [23] uses streaming;
 Batch Dataflow systems: DryadLINQ: high-level query language [28]; Spark uses cache [29, 30]; Dandelion extends to GPU and FPGA; Core problem: input data immutable;
 TF differs from batch dataflow in: (1) concurrent executions on overlapping subgraphs; (2) mutable state when training large model
 Dynamic control flow: like RNN; original dynamic dataflow[1] applies Switch and Merge;

# Parallel
1. weird trick to parallel, distributed (NIPS 2012);
2. Data parallel: eliminate IO bottleneck; multi-thread on single machine; distributed: DryadLINQ (OSDI 2008);
3. Distributed data parallel Dryad (ACM SIGOPS 2007);
4. Distributed data query: DryadLINQ [28]
5. Model parallel: shard models across (NLP);
6. Halide (SIGPLAN 2013): a language and compiler to optimize parallel;

# Linear Algebra
CPU:
 Eigen (with GPU support)
 Blas
 cuBLAS
 gemmlowp

GPU:
 Neon uses hand-optimized convolution, outperform TF;
 GPU-GPU communication: NVIDIA NCCL;
 cuDNN: conv, batch-norm, deconv, pooling, lstm, ...
 CNMeM: cuda-malloc, allocator

# Tensorboard
 1. “Name scope” to group nodes and subgraphs to blocks;
 2. Summary: track performance over time, scalar, histogram;

# Distributed
1. Across device: remove edges, replace with send and recv;
2. Fault tolerance:
 (a) Send/Receive, periodic check on TF;
 (b) Mesos or Borg cluster management: do not guarantee resource;
 (c) Strong consistency in Spark-RDD [30]: big overhead, small benefits;
 (d) SGD do not require strong consistency; TF: a Chubby [5] or Zookeeper [17] like system, maps task IDs to IP addresses;
 (e) backup worker: aggregates and take the first m of n updates; improve throughput in case of a straggler (slow);
 (f) Checkpoint: save and restore; no consistency required for async, can scheme after synchronous update
3. Parameter Server:
4. KV-store (key-value push/pull) on MXNet;
5. Mutable is crucial when training large models [6, 9, 20]
6. Consistency checking [20];
7. Elastic rescaling [20];
8. Scalability comes at the cost of big overheads [21];
9. Spark, SparkNet [30, 29];
10. Distribute over mobile devices: federated learning of DN by model averaging, Google;
11. Communication: multiple protocols supported;
 (a) gRPC over TCP;
 (b) RDMA over Converged Ethernet;
 (c) Optimized GPU-GPU communication;

 # Collective Communication
 Mainly used to pass gradients and parameters;
  1. Recursive halving and doubling algorithm (MPI);
  2. Bucket algorithm, or ring algorithm;

 # Server:
  Borg (Google)
  Big-Basin (Facebook)
