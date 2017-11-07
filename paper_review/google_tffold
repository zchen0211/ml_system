# Deep learning with dynamic computation graphs
 # ICLR 2017
  • Assign a depth to each node in the graph. Nodes with no dependencies (constants) are assigned depth zero. Nodes with only dependencies of depth zero are assigned depth one, nodes whose dependencies have a maximum depth of one get assigned depth two, etc.
  • Insert pass-through (identity) operations so that an operation at depth d + 1 only refers to results at depth d.
  • Batch together all nodes invoking the same operation at the same depth into a single node.
  • Concatenate all outputs which have the same depth and tensor type. The order of concatenation corresponds to the order in which the dynamic batching operations were enumerated.
  • Assign a label (d, t, i) to each edge in the original graph, where d is the depth, t is the tensor type, and i is the integer index for that edge into the (concatenated) outputs for d, t. The schedule for the graph consists of the indices i for all edges, which are grouped together by depth and operation.
