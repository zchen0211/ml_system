

Each node in a TensorFlow computational graph is called an *op*.  When we execute an op, TensorFlow runs its implementation, known as a *kernel*.  Consider ops as C++ function declarations, kernels are function definitions.  However, each op could have multiple kernels, for example, one calling cuDNN and running on GPUs and another calling MKL and running on Intel CPUs.

To define an op, we call macro [`REGISTER_OP`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op.h.html#294), which registers the name of the op and a protobuf message that describes the op, as well a C++ function that infers the shape of outputs given that of inputs.  

To define a kernel, we define a C++ class derived from [`OpKernel`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op_kernel.h.html#tensorflow::OpKernel), and call macro [`REGISTER_KERNEL_BUILDER`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op_kernel.h.html#_M/REGISTER_KERNEL_BUILDER) to associate the class to an op.

## Ops

Let's take the [definition of Abs Op](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/ops/math_ops.cc.html#152) as an example:

```cpp
REGISTER_OP("Abs")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {half, float, double, int32, int64}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Computes the absolute value of a tensor.

Given a tensor `x`, this operation returns a tensor containing the absolute
value of each element in `x`. For example, if x is an input element and y is
an output element, this operation computes \\(y = |x|\\).
)doc");
```

### `REGISTER_OP`

[`REGISTER_OP`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op.h.html#294) is defined as:

```cpp
#define REGISTER_OP(name) REGISTER_OP_UNIQ_HELPER(__COUNTER__, name)
#define REGISTER_OP_UNIQ_HELPER(ctr, name) REGISTER_OP_UNIQ(ctr, name)
#define REGISTER_OP_UNIQ(ctr, name)                                          \
  static ::tensorflow::register_op::OpDefBuilderReceiver register_op##ctr    \
      TF_ATTRIBUTE_UNUSED =                                                  \
          ::tensorflow::register_op::OpDefBuilderWrapper<SHOULD_REGISTER_OP( \
              name)>(name)
```

It calls `OpDefBuilderWrapper`'s constructor.  Then all subsequent calls to `.Input`, `.Output`, etc. just calls `OpDefBuilder`'s corresponding methods and return `*this`.  These methods parse Op definitions into an `OpDefBuilder` typed variable, which is then assigned to an `OpDefBuilderReceiver` typed variable.  The latter's constructor registers the definition.


### Selective Registration

In above code snippet, [`SHOULD_REGISTER_OP(name)`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/selective_registration.h.html) is `true` by default.  However, if we pass `-DSELECTIVE_RGISTRATION` to GCC, only Ops declared in a file called `ops_to_register.h` are registered and will be linked.  This approach limits the size of the generated binary file.

The class template [`OpDefBuilderWrapper`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op.h.html#tensorflow::register_op::OpDefBuilderWrapper) accepts a template parameter `should_register`.  The specialization with `should_register=true` contains a data member of type `OpDefBuilder`, and all its methods calls methods of `OpDefBuiilder`.  The specialization with `should_register=false` had the same set of methods but defined as no-ops.

### `OpDefBuilder`

[`OpDefBuilder`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op_def_builder.h.html#tensorflow::OpDefBuilder)'s methods process their inputs into its `OpRegistrationData` typed data member [`op_reg_data_`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op_def_builder.h.html#tensorflow::OpDefBuilder::op_reg_data_).

[`OpRegistrationData`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op_def_builder.h.html#tensorflow::OpRegistrationData) is a C++ struct:

```cpp
struct OpRegistrationData {
 public:
  OpRegistrationData() {}
  OpRegistrationData(const OpDef& def) : op_def(def) {}

  OpDef op_def;
  OpShapeInferenceFn shape_inference_fn;
};
```

where [`OpDef`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def.proto) is a protobuf message of

```proto
optional string name = 1;
repeated .tensorflow.OpDef.ArgDef input_arg = 2;
repeated .tensorflow.OpDef.ArgDef output_arg = 3;
repeated .tensorflow.OpDef.AttrDef attr = 4;
...
```

[`OpShapeInferenceFn`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op_def_builder.h.html#tensorflow::OpShapeInferenceFn) is defined as:

```cpp
typedef std::function<Status(shape_inference::InferenceContext* c)> OpShapeInferenceFn;
```

### `OpDefBuilderReceiver`

Struct [`OpDefBuilderReceiver`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op.h.html#tensorflow::register_op::OpDefBuilderReceiver) has only two constructors.  One accepts an `OpDefBulderWrapper<false>` and does nothing, [the other one](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op.cc.html#_ZN10tensorflow11register_op20OpDefBuilderReceiverC1ERKNS0_19OpDefBuilderWrapperILb1EEE) accepts an `OpDefBuilderWrapper<true>` and registers the finalized `OpDefBuilder`:

```cpp
OpDefBuilderReceiver::OpDefBuilderReceiver(
    const OpDefBuilderWrapper<true>& wrapper) {
  OpRegistry::Global()->Register(
      [wrapper](OpRegistrationData* op_reg_data) -> Status {
        return wrapper.builder().Finalize(op_reg_data);
      });
}
```

[`OpRegistry::Global()`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op.cc.html#_ZN10tensorflow10OpRegistry6GlobalEv) just returns a static variable:

```cpp
OpRegistry* OpRegistry::Global() {
  static OpRegistry* global_op_registry = new OpRegistry;
  return global_op_registry;
}
```

### `OpRegistry` and `OpRegistryInterface`

[`OpRegistry`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op.h.html#tensorflow::OpRegistry) implements [`OpRegistryInterface`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op.h.html#tensorflow::OpRegistryInterface), which defines two methods:

1. LookUp, which returns the `OpRegistrationData` given op's name,
1. LookUpOpDef, which returns proto message `OpDef`, which is a data member of `OpRegistrationData`. [This looks duplicated.]

`OpRegistry::Register` accepts a lambda and passes it to [`OpRegistry::RegisterAlreadyLocked`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op.cc.html#_ZNK10tensorflow10OpRegistry21RegisterAlreadyLockedESt8functionIFNS_6StatusEPNS_18OpRegistrationDataEEE).  The latter 

1. creates an `OpRegistrationData` object and calls the lambda to fill it by calling [`OpDefBuilder::Finalize`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op_def_builder.cc.html#_ZNK10tensorflow12OpDefBuilder8FinalizeEPNS_18OpRegistrationDataE), 
1. calls [`ValidateOpDef`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op_def_util.cc.html#_ZN10tensorflow13ValidateOpDefERKNS_5OpDefE),
1. regsiter the `OpRegistrationData` object to [`OpRegistry::registry_`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op.h.html#tensorflow::OpRegistry::registry_).
1. calls the callback function [`OpRegistry::watcher_`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op.h.html#tensorflow::OpRegistry::watcher_), if there is one.

### Conclusion

The core part of above framework is the parsing of Op definitions, which is in `OpDefBuilder`'s methods [`Input`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op_def_builder.cc.html#_ZN10tensorflow12OpDefBuilder5InputENS_11StringPieceE), [`Output`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op_def_builder.cc.html#_ZN10tensorflow12OpDefBuilder6OutputENS_11StringPieceE), [`Attr`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op_def_builder.cc.html#_ZN10tensorflow12OpDefBuilder4AttrENS_11StringPieceE), etc.  From the example at the beginning of this article, we can see that these methods take descriptions in a special syntax.  So the implementation of `OpDefBuilder` is largely a parser.  A key data type used by this parser is [StringPiece](https://github.com/PaddlePaddle/Paddle/pull/2432).

## Kernels

This article has been too long.  I wrote this section in a [separate article](TensorFlow-Kernels).
