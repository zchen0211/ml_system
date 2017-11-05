### 概述
tensorflow是一个分布式神经网络框架。它基于计算图来描述计算过程，提供了完善而灵活的分布式支持，既方便研究和做实验，工程性能上也不太差。

### 计算图描述
A TensorFlow computation is described by a directed graph, which is composed of a set of nodes. The graph represents a dataflow computation.

An operation has a name and represents an abstract computation (e.g., “matrix multiply”, or “add”). 

计算图相关信息用proto描述，下面是其定义：
```shell
message GraphDef {
	repeated NodeDef node = 1;
	FunctionDefLibrary library = 2;
	int32 version = 3;
}
NodeDef {
	string name = 1;
	string op = 2;
	repeated string input = 3;
	string device = 4;
	map<string, AttrValue> attr = 5;
}
```
Node中包含计算op，数据inputs和设备信息device(cpu or gpu)。

### tensor
header file(https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.h)

和其他的神经网络框架类似，tensor用来描述一个多维数组(A tensor simply identifies a multidimensional array or list)，主要有三个属性Ranks, Shapes, and Types(https://www.tensorflow.org/programmers_guide/dims_types)。

tensorflow的tensor主要基于Eigen::Tensor并且做了大量的扩展。
引用的eigen文件(https://github.com/RLovelett/eigen/blob/master/unsupported/Eigen/CXX11/Tensor)

从构造函数可以看出起主要成员：
```cpp
  /// \brief Creates a tensor with the input `type` and `shape`, using
  /// the allocator `a` and the specified "allocation_attr" to
  /// allocate the underlying buffer. 
  Tensor(Allocator* a, DataType type, const TensorShape& shape,
         const AllocationAttributes& allocation_attr);
```
tensorflow的tensor可以通过Allocator来分配和管理buffer。(https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/allocator.h#L65)
```shell
Allocator is an abstract interface for allocating and deallocating device memory.
```

### operator and opkernel.
refs: (https://www.tensorflow.org/extend/adding_an_op)

一个tensorflow的op主要包含两个部分：
#### 1， op interface. 
主要用于描述op的输入输出等性质，并且负责注册到tensorflow的系统中。下面这段代码注册了一个叫做ZeroOut的op，并且描述了其输入类型是32位int型，输出也是32位int型，并且对其shape做了描述，输入输出的tensor shape是一样的。
```cpp
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
```

#### 2，op kernel的实现。
主要是要继承OpKernel这个基类，并且实现`Compute`这个接口，Compute有一个输入参数OpKernelContext，输入输出都是通过这个context进行管理。
```cpp
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output(0) = input(0);
  }
};
```
同样的，这个kernel需要注册到tensorflow的系统中。
```cpp
REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
```

### 分布式实现
refs: (https://www.tensorflow.org/extend/architecture)

tensorflow对分布式的支持还是很灵活和完善的，不过因为暴露的接口过于底层，难以理解和上手。

在tensorflow分布式系统中，主要有三个角色：client，master和worker。

#### Client
client的主要作用：
1. construct graph，构建计算图。
2. uses the Session to communicate with the master. 用Session来和master通信，管理计算图的分发和计算资源。
3. runStep. 驱动计算step。

#### Master
master的主要作用：
1. create subgraph for each device(one per device). 负责将client构建好的graph切分成子graph，并且添加相应的通信节点(多机)。
2. device placement. Master需要把切分好的子计算graph分配到不同的计算设备上。
3. register/run subgraph. 驱动各个设备运行各自分到的subgraph。

### Worker
worker的主要作用: 
1. access to one or more computational devices (such as CPU cores or GPU cards) 
2. execute graph nodes on those devices as instructed by the master. 


### Gradient计算
https://www.tensorflow.org/versions/r0.11/api_docs/python/train/gradient_computation

https://www.tensorflow.org/api_guides/python/train#gradient_computation
TensorFlow provides functions to compute the derivatives for a given TensorFlow computation graph, adding operations to the graph. The optimizer classes automatically compute derivatives on your graph.

When TensorFlow needs to compute the gradient of a tensor C with respect to some tensor I on which C depends, it first finds the path in the computation graph from I to C. Then it backtracks from C to I, and for each operation on the backward path it adds a node to the TensorFlow graph, composing the partial gradients along the backwards path using the chain rule. The newly added node computes the “gradient function” for the cor- responding operation in the forward path. A gradient function may be registered by any operation. This func- tion takes as input not only the partial gradients com- puted already along the backward path, but also, option- ally, the inputs and outputs of the forward operation.

### 疑问：
1. optimizer如何表达？
2. 反向传播如何实现？
3. protobuf表达的好坏？
4. tensor和eigen如何结合的？


