# Caffe2's Operator

Caffe2 has a concept *operator*, which corresponds to TensorFlow's *Op*.

Different from Op, an operator usualy accompany with a *gradient operator* (GradientOp).

Let us take [`ReluOp`](http://yiwang.ngrok.io/codebrowser/caffe2/operators/relu_op.h.html#caffe2::ReluOp) and [`ReluGradientOp`](http://yiwang.ngrok.io/codebrowser/caffe2/operators/relu_op.h.html#caffe2::ReluGradientOp) as an example.

## Operator Definition

All operators are classes derived from [`Operaotr<Context>`](http://yiwang.ngrok.io/codebrowser/caffe2/core/operator.h.html#caffe2::Operator).

```cpp
template <typename T, class Context>
class ReluOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(ReluOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;
};

template <typename T, class Context>
class ReluGradientOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(ReluGradientOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;
};
```

## `Operator`

`Operator<Context>` has a data member `Context context_`, which records the current device (or GPU).  The constructor of `Operator` initializes `context_` by passing in its constructor a proto message [`OperatorDef`](http://yiwang.ngrok.io/codebrowser/build/caffe2/proto/caffe2.pb.h.html#caffe2::OperatorDef).  Then `Operator::Operator` calls `context_.SwitchToDevice(0)`.


`Operator<Context` has three virtual functions:

1. `RunOnDevice() = 0` is what you want to override,
1. `Run(stream_id)` calls `context_.SwitchToDevice(stream_id)`,  `RunOnDevice`, and `context_.FinishDeviceComputation`, and
1. `RunAsync` calls `context_.SwitchToDevice(stream_id)` and  `RunOnDevice`.

[TODO: Check what `Context::FinishDeviceComputation` does.]


`Operator<Context>` also allows user overriden `RunOnDevice` to access inputs and outputs through:

- [`const Tensor<Context>& Input(idx)`](http://yiwang.ngrok.io/codebrowser/caffe2/core/operator.h.html#_ZN6caffe28Operator5InputEi), and
- [`Tensor<Context>* Output(idx)`](http://yiwang.ngrok.io/codebrowser/caffe2/core/operator.h.html#_ZN6caffe28Operator6OutputEi)

## `OperatorBase`

`Operator<Context>` derives from class [`OperatorBase`](http://yiwang.ngrok.io/codebrowser/caffe2/core/operator.h.html#caffe2::OperatorBase).

