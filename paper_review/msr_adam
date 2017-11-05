# Project Adam: Building an Efficient and Scalable Deep Learning Training System
 OSDI 2014

 no locks (Hogwild)
 120 machines, on CPU
 pass pointer rather than data (built on Windows socket, port)
 working set: L3 cache
 slow fast machine: 75% finished (fast machine does not steal work from slow ones)
 PS: sometimes send data and compute gradient on the server sides
  memory usage of FC: O(m*n) -> O(k*(m+n))

  throughput optimization: NUMA aware
   SSE/AVX instructions for applying updates and all processing
  Paxos

 Exp:
  MNIST
  ImageNet
