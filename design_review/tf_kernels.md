In the [previous article](TensorFlow-Ops), we discussed TensorFlow ops.  In this article we review kernels.

## Kernels

Again, let us start from a real example, the [Abs kernel](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/kernels/cwise_op_abs.cc.html#31):

```cpp
REGISTER5(UnaryOp, CPU, "Abs", functor::abs, float, Eigen::half, double, int32,
          int64);
```

where macro [`REGISTER5`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/kernels/cwise_ops_common.h.html#440) means calling [`REGISTER`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/kernels/cwise_ops_common.h.html#_M/REGISTER) for 5 times. And `REGISTER` calls [`REGISTER_KERNEL_BUILDER`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op_kernel.h.html#_M/REGISTER_KERNEL_BUILDER):

```cpp
#define REGISTER(OP, D, N, F, T)                                             \
  REGISTER_KERNEL_BUILDER(Name(N).Device(DEVICE_##D).TypeConstraint<T>("T"), \
                          OP<D##Device, F<T>>);
```

So above call to `REGISTER5` expands to

```cpp
REGISTER_KERNEL_BUILDER(Name("Abs").Device(DEVICE_CPU).TypeConstraint<float>("float"), UnaryOp<CPUDevice, functor::abs<float>>);
REGISTER_KERNEL_BUILDER(Name("Abs").Device(DEVICE_CPU).TypeConstraint<Eigen::half>("Eigen::half"), UnaryOp<CPUDevice, functor::abs<Eigen::half>>);
REGISTER_KERNEL_BUILDER(Name("Abs").Device(DEVICE_CPU).TypeConstraint<double>("double"), UnaryOp<CPUDevice, functor::abs<double>>);
REGISTER_KERNEL_BUILDER(Name("Abs").Device(DEVICE_CPU).TypeConstraint<int32>("int32"), UnaryOp<CPUDevice, functor::abs<int32>>);
REGISTER_KERNEL_BUILDER(Name("Abs").Device(DEVICE_CPU).TypeConstraint<int64>("int64"), UnaryOp<CPUDevice, functor::abs<int64>>);
```

There is another macro invocation for registering  GPU versions of Abs:

```cpp
REGISTER4(UnaryOp, GPU, "Abs", functor::abs, float, Eigen::half, double, int64);
```

which expands to:

```cpp
REGISTER_KERNEL_BUILDER(Name("Abs").Device(DEVICE_GPU).TypeConstraint<float>("float"), UnaryOp<GPUDevice, functor::abs<float>>);
REGISTER_KERNEL_BUILDER(Name("Abs").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("Eigen::half"), UnaryOp<GPUDevice, functor::abs<Eigen::half>>);
REGISTER_KERNEL_BUILDER(Name("Abs").Device(DEVICE_GPU).TypeConstraint<double>("double"), UnaryOp<GPUDevice, functor::abs<double>>);
REGISTER_KERNEL_BUILDER(Name("Abs").Device(DEVICE_GPU).TypeConstraint<int64>("int64"), UnaryOp<GPUDevice, functor::abs<int64>>);
```

Each line registers a specialization of class template `UnaryOp` as a kernel version for a specified device and numeric type.

### `REGISTER_KERNEL_BUILDER`

[`REGISTER_KERNEL_BUILDER`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op_kernel.h.html#_M/REGISTER_KERNEL_BUILDER) is defined as:

```cpp
#define REGISTER_KERNEL_BUILDER(kernel_builder, ...) \
  REGISTER_KERNEL_BUILDER_UNIQ_HELPER(__COUNTER__, kernel_builder, __VA_ARGS__)

#define REGISTER_KERNEL_BUILDER_UNIQ_HELPER(ctr, kernel_builder, ...) \
  REGISTER_KERNEL_BUILDER_UNIQ(ctr, kernel_builder, __VA_ARGS__)

#define REGISTER_KERNEL_BUILDER_UNIQ(ctr, kernel_builder, ...)          \
  static ::tensorflow::kernel_factory::OpKernelRegistrar                \
      registrar__body__##ctr##__object(                                 \
          SHOULD_REGISTER_OP_KERNEL(#__VA_ARGS__)                       \
              ? ::tensorflow::register_kernel::kernel_builder.Build()   \
              : nullptr,                                                \
          #__VA_ARGS__, [](::tensorflow::OpKernelConstruction* context) \
                            -> ::tensorflow::OpKernel* {                \
                              return new __VA_ARGS__(context);          \
                            });
```

The introduction of `REGISTER_KERNEL_BUILDER_UNIQ_HELPER` and `REGISTER_KERNEL_BUILDER_UNIQ` are to use `__COUNTER__`, a pre-defined macro provided by [GCC](https://gcc.gnu.org/onlinedocs/cpp/Common-Predefined-Macros.html) and Visual C++:

> This macro expands to sequential integral values starting from 0. In conjunction with the ## operator, this provides a convenient means to generate unique identifiers. Care must be taken to ensure that __COUNTER__ is not expanded prior to inclusion of precompiled headers which use it. Otherwise, the precompiled headers will not be used.

From above macro definitions, we see that it's the constructor of [`::tensorflow::kernel_factory::OpKernelRegistrar`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op_kernel.h.html#tensorflow::kernel_factory::OpKernelRegistrar) who registers a kernel.  This constructor's definition is:

```cpp
  typedef OpKernel* (*Factory)(OpKernelConstruction*);
  OpKernelRegistrar(const KernelDef* kernel_def, StringPiece kernel_class_name,
                    Factory factory) {
```

And one of above variant of Abs expands to 

```cpp
static ::tensorflow::kernel_factory::OpKernelRegistrar registrar__body__4__object(
    true ?
    ::tensorflow::register_kernel::Name("Abs").Device(DEVICE_CPU).TypeConstraint<int64>("T").Build() :
    nullptr,
    "UnaryOp< CPUDevice, functor::abs<int64>>",
    [](::tensorflow::OpKernelConstruction* context) -> ::tensorflow::OpKernel* {
      return new UnaryOp< CPUDevice, functor::abs<int64>>(context);
    });
```

### `KernelDef` and `KernelDefBuilder`

Above Abs example creates the first parameter of type `KernelDef` by calling the constructor of [`Name`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op_kernel.h.html#_ZN10tensorflow15register_kernel4NameC1EPKc).  `Name` is a sub-class of [`KernelDefBuilder`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/kernel_def_builder.h.html#tensorflow::KernelDefBuilder), whose [`Build` method](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/kernel_def_builder.h.html#_ZN10tensorflow16KernelDefBuilder5BuildEv) returns the address of data member [`KernelDef KernelDefBuilder::kernel_def_`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/kernel_def_builder.h.html#tensorflow::KernelDefBuilder::kernel_def_).  [`KernelDef`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/contrib/cmake/tensorflow/core/framework/kernel_def.pb.h.html#tensorflow::KernelDef) is a protobuf message.

[`KernelDefBuilder`'s constructor](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/kernel_def_builder.cc.html#_ZN10tensorflow16KernelDefBuilderC1EPKc) fills in `KernelDef::op_name`.

[`KernelDefBuilder::Device`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/kernel_def_builder.cc.html#_ZN10tensorflow16KernelDefBuilder6DeviceEPKc) fills in `KernelDef::device_type`.

[`KernelDefBuilder::TypeConstraint`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/kernel_def_builder.h.html#_ZN10tensorflow16KernelDefBuilder14TypeConstraintEPKc) is a method template, whose definition is as follows:

```cpp
template <class T>
KernelDefBuilder& KernelDefBuilder::TypeConstraint(const char* attr_name) {
  return this->TypeConstraint(attr_name, DataTypeToEnum<T>::v());
}
```

where [`DataTypeToEnum`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/types.h.html#tensorflow::DataTypeToEnum) is a class template, whose each specialization maps a C++ type to an enum ID.

`OpKernelRegistrar`'s constructor calls [`OpKernelRegistrar::InitInternal`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op_kernel.cc.html#tensorflow::KernelRegistration) to register its three parameters to a [singleton registry](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op_kernel.cc.html#tensorflow::KernelRegistration).  The third parameter is a C++ lambda, which, if called, allocates and returns a kernel object.

### Selective Registration

Note that [`SHOULD_REGISTER_OP_KERNEL`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/selective_registration.h.html#40) defines a selective registration mechanism like [`SHOULD_REGISTER_OP`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/selective_registration.h.html#44) does, as we explained in the previous article.  `SHOULD_REGISTER_OP_KERNEL` defaults to `true`, unless `-DSELECTIVE_REGISTRATION` is given to GCC, where only classes whose names are listed in variable `kNecessaryOpKernelClasses` in header file `ops_to_register.h` would be registered.

### Kernel Classes

In above Abs example, the registered kernel class is `UnaryOp< CPUDevice, functor::abs<int64>>`, where [`UnaryOp`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/kernels/cwise_ops_common.h.html#tensorflow::UnaryOp) is a sub-class of [`OpKernel`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op_kernel.h.html#tensorflow::OpKernel).  All kernels are classes derived from `OpKernel`.

The first template parameter of `UnaryOp` can take the value of either [`CPUDevice`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/kernels/cwise_ops_gradients.h.html#tensorflow::functor::CPUDevice):

```cpp
typedef Eigen::ThreadPoolDevice CPUDevice;
```

or `GPUDevice`:

```cpp
typedef Eigen::GpuDevice GPUDevice;
```

The other template parameter of `UnaryOp` is a functor.  In above example, it's [`abs`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/kernels/cwise_ops.h.html#tensorflow::functor::abs):

```cpp
template <typename T>
struct abs : base<T, Eigen::internal::scalar_abs_op<T>,
                  typename Eigen::internal::scalar_abs_op<T>::result_type> {};
```

where `base` defines some types:

```cpp
template <typename T, typename F, typename R = T>
struct base {
  typedef F func;
  typedef R out_type;
  typedef T in_type;
  typedef typename TTypes<out_type>::Flat tout_type;
  typedef typename TTypes<in_type>::ConstFlat tin_type;
  typedef typename TTypes<in_type>::ConstScalar tscalar_type;
  ...
};
```

These two template parameters are used to implement [`UnaryOp::Compute`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/kernels/cwise_ops_common.h.html#_ZN10tensorflow7UnaryOp7ComputeEPNS_15OpKernelContextE):

```cpp
  void Compute(OpKernelContext* ctx) override {
    const Tensor& inp = ctx->input(0);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, inp.shape(), &out));
    functor::UnaryFunctor<Device, Functor>()(
        ctx->eigen_device<Device>(), out->flat<Tout>(), inp.flat<Tin>());
  }
```

where [`UnaryFunctor`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/kernels/cwise_ops_common.h.html#tensorflow::functor::UnaryFunctor) executes the `abs` functor:

```cpp
template <typename Functor>
struct UnaryFunctor<CPUDevice, Functor> {
  void operator()(const CPUDevice& d, 
                  typename Functor::tout_type out,
                  typename Functor::tin_type in) {
    Assign(d, out, in.unaryExpr(typename Functor::func()));
  }
};
```

Pleae be aware that the `Functor::func` here refers to `base::func`, which, in this example, is `Eigen::internal::scalar_abs_op<T>`.  So `UnaryOp::Compute` actually calls `Eigen::internal::scalar_abs_op<T>` to compute the abs value.

Template funcion [`Assign`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/kernels/cwise_ops_common.h.html#_ZN10tensorflow7functor6AssignERKT_T0_T1_) assigns the result to `out`:

```cpp
template <typename D, typename Out, typename Rhs>
void Assign(const D& d, Out out, Rhs rhs) {
  out.device(d) = rhs;
}
```

### Kernel Execution

Note that `Compute` takes a parameter [`OpKernelContext`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op_kernel.h.html#tensorflow::OpKernelContext)`* ctx`, which provide references to resource managers that allocate resources at graph execution time.

We will review the complete process of kernel execution in a subsequent article.

### Kernel Creation

Before executing a kernel, TensorFlow needs to create it by calling the registered factory lambda:

```cpp
    [](::tensorflow::OpKernelConstruction* context) -> ::tensorflow::OpKernel* {
      return new UnaryOp< CPUDevice, functor::abs<int64>>(context);
    }
```    

which in turn calls [the constructor of `UnaryOp`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/numeric_op.h.html#_ZN10tensorflow7UnaryOpC1EPNS_20OpKernelConstructionE):

```cpp
template <class T>
class BinaryOp : public OpKernel {
 public:
  explicit UnaryOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({dt}, {dt}));
  }
```

where [`OP_REQUIRES_OK`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/op_kernel.h.html#_M/OP_REQUIRES_OK) checks that the data type specified in class template parameter matches the one in `OpKernelConstruction` parameter passed in by TensorFlow framework at graph creation time.

### Conclusion

I will write another article to detail the definition, creation, and execution of a graph.
