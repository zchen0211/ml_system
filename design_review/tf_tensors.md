# TenorFlow's Tensor

[`tensorflow::Tensor`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/tensor.h.html#tensorflow::Tensor) represents a n-dimensional array of values, like `caffe2::Tensor`.

Different from `caffe2::Tensor<Context>`, which is a template, `tesnorflow::Tensor` is a class.

`caffe2::Tensor<Context>`'s constructor doesn't allocate memory; instead, memory allocate is delayed till the `mutable_data` is called.  Whereas `tensorflow::Tensor` [allocates the memory](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/tensor.cc.html#321).

`caffe2::Tensor<Context>`'s template methods `data<T>` and `mutalbe_data<T>` can return an array of any typed elements -- `caffe2::Tensor::meta_` records the most recently returned (and allocated) element type.  Whereas `tensorflow::Tensor`'s constructor accepts a [`DataType`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/contrib/cmake/tensorflow/core/framework/types.pb.h.html#tensorflow::DataType) typed parameter that specifies the element type.

`caffe2::Tensor<Context>` supports only numerical typed elements.  Whereas `tensorflow::Tensor` supports [`string`-typed elements](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/tensor.cc.html#tensorflow::(anonymousnamespace)::Helper).

`caffe2::Tensor<Context>` doesn't support accessing data in protobuf messages.  Whereas `tensorflow::Tensor` [does](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/tensor.cc.html#tensorflow::(anonymousnamespace)::ProtoHelper).

`caffe2::Tensor<Context>`'s destructor doesn't free memory; instead, its data member `shared_ptr<T> data_` does.  Whereas `tensorflow::Tensor`'s destructor takes the responsibility to free memory.   In addition, `tensorflow::Tensor` counts the reference of the memory by itself, whereas `caffe2::Tensor<Context>` utilizes `shared_ptr` for that.

## `TensorShape`

The shape of a tensor is represented by [`tensorflow::TensorShape`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/core/framework/tensor_shape.h.html#tensorflow::TensorShape), which can be constructed from a list of `int64` values, or from a protobuf message [`TensorShapeProto`](http://yangff.coding.me/tf-doc/codebrowser/codebrowser/tensorflow/contrib/cmake/tensorflow/core/framework/tensor_shape.pb.h.html#tensorflow::TensorShapeProto).

`TensorShape` supports various representations of a shape because most tensors are low dimensional.  This brings more complexity than Caffe2's `vector<int64_t>`.  Indeed, `tensor_shape.h` and `tensor_shape.cc` take 759 lines of C++ code in total -- more than the very candy `majel::Dim` that takes 498 lines.


## Memory Management

The [constructor](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/framework/tensor.cc#L580) of `tensorflow::Tensor` accepts a parameter `Allocator* a` and passes it to a newly created [`tensorflow::Buffer`](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/framework/tensor.cc#L87) object `tensorflow::Tensor::buf_`:

```cpp
Tensor::Tensor(Allocator* a, DataType type, const TensorShape& shape)
    : shape_(shape), buf_(nullptr) {
  set_dtype(type);
  CHECK_NOTNULL(a);
  if (shape_.num_elements() > 0 || a->ShouldAllocateEmptyTensors()) {
    CASES(type, buf_ = new Buffer<T>(a, shape.num_elements()));
  }
```


`tensorflow::Buffer` then saves `a` into its parent class [`tensorflow::BufferBase`](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/framework/tensor.cc#L52)'s `alloc_` field, and it calls `Allocator::Allocate<T>`:

```cpp
template <typename T>
Buffer<T>::Buffer(Allocator* a, int64 n)
    : BufferBase(a), data_(a->Allocate<T>(n)), elem_(n) {}
```	

[`Allocator::Allocate<T>`](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/framework/allocator.h#L109) calls [`Allocator::AllocateRaw`](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/framework/allocator.h#L92) and then call type `T`'s constructors via `Allocator::RunCtor<T>`:

```cpp
  template <typename T>
  T* Allocate(size_t num_elements,
              const AllocationAttributes& allocation_attr) {
    ...
    void* p = AllocateRaw(kAllocatorAlignment, sizeof(T) * num_elements,
                          allocation_attr);
    T* typed_p = reinterpret_cast<T*>(p);
    if (typed_p) RunCtor<T>(typed_p, num_elements);
    return typed_p;
  }
```

By default, `Allocator::RunCtor<T>` is an no-op, so it doesn't construct basic types.  A [specialization](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/framework/allocator.h#L239) runs string type's constructor:

```cpp
template <>
inline void Allocator::RunCtor(string* p, size_t n) {
  RunStringCtor(p, n);
}
```

Similarly, there are corresponding `Allocator::RunDtor<T>` defines.

[`Allocator::AllocateRaw`](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/framework/allocator.cc#L71) calls `port::AlignedMalloc`:

```cpp
  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    void* p = port::AlignedMalloc(num_bytes, alignment);
    ...
    return p;
  }
```

and [`Allocator::DeallocateRaw`](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/framework/allocator.cc#L86) calls `port::AlignedFree`:

```cpp
  void DeallocateRaw(void* ptr) override {
    ...
    port::AlignedFree(ptr);
  }
```


`port:AlignedMalloc`, `port::AlignedFree`, and other platform-independent memory allocation are in [tensorflow/core/platform/mem.h](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/platform/mem.h):

```cpp
namespace tensorflow {
namespace port {

void* AlignedMalloc(size_t size, int minimum_alignment);
void AlignedFree(void* aligned_memory);

void* Malloc(size_t size);
void* Realloc(void* ptr, size_t size);
void Free(void* ptr);

}
}
```


There are two implemntations:

1. POSIX implemenation in [tensorflow/core/platform/posix/port.cc](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/platform/posix/port.cc) just calls POSIX C-runtime functions like malloc.  For example:

```cpp
void* Malloc(size_t size) {
#ifdef TENSORFLOW_USE_JEMALLOC
  return jemalloc_malloc(size);
#else
  return malloc(size);
#endif
}
```

1. Windows implementation in [tensorflow/core/platform/windows/port.cc](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/platform/windows/port.cc) is almost identical with the POSIX one, because the C-runtime functions are almost the same.


### Question: GPU Memory

Above two implementation both allocates CPU memory, but not GPU memory.

TensorFlow codebase doesn't call `cudaMalloc`. Instead, there is one function,  [`perftools::gputools::cuda::CUDADriver::DeviceAllocate`](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/stream_executor/cuda/cuda_driver.h#L103), that calls `cuMemAlloc`:

```cpp
/* static */ void *CUDADriver::DeviceAllocate(CudaContext *context,
                                              uint64 bytes) {
  ...
  CUresult res = cuMemAlloc(&result, bytes);
```

[Class `CUDADriver`](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/stream_executor/cuda/cuda_driver.h#L59) includes a set of static methods, each corresponds to a CUDA API.  For example, [`CUDADriver::DeviceDeallocate`](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/stream_executor/cuda/cuda_driver.cc#L935) calls `cuMemFree`:

```cpp
/* static */ void CUDADriver::DeviceDeallocate(CudaContext* context,
                                               void *location) {
  ...
  CUresult res = cuMemFree(pointer);
```


Only [`CUDAExecutor::Allocate(uint64 size)`](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/stream_executor/cuda/cuda_gpu_executor.cc#L433) calls `CUDADriver::DeviceAllocate(context_, size)`:

```cpp
void *CUDAExecutor::Allocate(uint64 size) {
  return CUDADriver::DeviceAllocate(context_, size);
}
```

And I haven't figured it out how/if Tensor calls `CUDAExecutor::Allocate` for GPU memory.

