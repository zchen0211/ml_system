# `Caffe2::Tensor`


Caffe2 has [`caffe2::Tensor`](http://yiwang.ngrok.io/codebrowser/caffe2/core/tensor.h.html#caffe2::Tensor).  The purpose of `caffe2::Tensor` is to manage the memory of a tensor.  Users calls `data()` or `mutable_data()` to access the managed memory.  `data` and `mutable_data` can return a big enough block of any typed elements -- please go on reading for details.

### Template Parameters and Data Members

`caffe2::Tensor` has the following [data members](http://yiwang.ngrok.io/codebrowser/caffe2/core/tensor.h.html#caffe2::Tensor::dims_):

1. `vector<TIndex> dims_` records the size of each dimension.
1. `TIndex size_ = -1` is the total number of elements in this tensor.
1. `TypeMeta meta_`  contains constructor, destructor, and copy lamda of a certain element type.
1. `std::shared_ptr<void> data_` holds the data.
1. `bool shares_data_ = false` marks if this tensor shares data with another tensor.
1. `size_t capacity_ = 0` in case of chunk load we store how much data was already loaded.

We can see that `caffe2::Tensor` uses a [`vector<TIndex>`](http://yiwang.ngrok.io/codebrowser/caffe2/core/tensor.h.html#_ZN6caffe26TensorC1ERKNSt3__16vectorIxNS1_9allocatorIxEEEE) to represent the size of tensors, where `typedef int64_t TIndex;`.

In addition to data members, `caffe2::Tensor` takes a template parameter, `Context`, which could be `GPUContext` or `CPUContext`, that provides methods that allocate and delete memory.


### Memory Management Using `share_ptr`

Please be aware that the memory is owned by `std::shared_ptr<void> data_`, and `shared_ptr` takes not only a pointer to the data, and optionally a lambda that frees the memory.  `shared_ptr` calls the memory free lambda when the reference count goes down to zero.

As `caffe2::Tensor` uses void-typed `shared_ptr` (`shared_ptr<void> data_`), it requires the customized memory deletion functions -- the template parameter  `Context` provides them.  Also, `MetaType` provides ways to initialize allocated memory.

`meta_` either comes from other tensors copied by [copy-from-other-tensors constructors](http://yiwang.ngrok.io/codebrowser/caffe2/core/tensor.h.html#_ZN6caffe26TensorC1ERKNSt3__16vectorIxNS1_9allocatorIxEEEERKNS2_IT_NS3_IS8_EEEEPT_), or is created by `TypeMeta::Make<T>` from element type `T` by [copy-from-vector constructors](http://yiwang.ngrok.io/codebrowser/caffe2/core/tensor.h.html#_ZN6caffe26TensorC1ERKNSt3__16vectorIxNS1_9allocatorIxEEEERKNS2_IT_NS3_IS8_EEEEPT_).

### No Destructors

`caffe2::Tensor` doesn't have destructors.  `shared_ptr` is in charge of deleting the memory.


### Delayed Memory Allocation

`caffe2::Tensor`'s constructor calls [`Resize`](http://yiwang.ngrok.io/codebrowser/caffe2/core/tensor.h.html#_ZN6caffe26Tensor6ResizeEDpT_), which records the size into `dims_` and `size_` by calling `SetDims`, but doesn't allocate the memory.  The memory allocation is delayed until [`caffe2::Tensor::mutable_data()`](http://yiwang.ngrok.io/codebrowser/caffe2/core/tensor.h.html#_ZN6caffe26Tensor16raw_mutable_dataEv) is called.

To be more specific,  `Resize` checks if `capacity_ < size_ * meta_.itemize()`.

```cpp
  void Resize(Ts... dim_source) {
    bool size_changed = SetDims(dim_source...);
    // If needed, we will free the data. the next mutable_data() call
    // will create the data storage.
    if (size_changed && (capacity_ < size_ * meta_.itemsize() ||
                         !FLAGS_caffe2_keep_on_shrink)) {
      data_.reset();
      capacity_ = 0;
    }
  }
```

If so, it resets `data_`, and a later call to `mutable_data` will re-allocate the memory; otherwise, `Resize` does nothing, as there is sufficient memory and data accessors could work.


### Tensors of Any Type of Elements

It seems it doesn't hurt anyway if `Resize` re-allocate memory in case capacity is not enough.  Why does it leave the work to `mutable_data`?

The answer lies in the implementation of `mutable_data`:

```cpp
 template <typename T>
 inline T* mutable_data() {
   if ((size_ == 0 || data_.get()) && IsType<T>()) {
     return static_cast<T*>(data_.get());
   }
   return static_cast<T*>(raw_mutable_data(TypeMeta::Make<T>()));
 }
```

We can see that as a function template, `mutable_data` can return an array of any typed elements.  As a public method, `mutable_data` is supposed to be call by the client directly; it makes `Tensor` looks can hold any typed elements.

In above code, [`Tensor::IsType<T>()`](http://yiwang.ngrok.io/codebrowser/caffe2/core/tensor.h.html#_ZNK6caffe26Tensor6IsTypeEv) checks if the expected return type `T` is the current type recorded by `mata_`:

```cpp
 template <typename T>
 inline bool IsType() const { return meta_.Match<T>(); }
```

If they are the same, `mutable_data` returns the data pointer; otherwise, `mutable_data` calls [`raw_mutable_data`](http://yiwang.ngrok.io/codebrowser/caffe2/core/tensor.h.html#_ZN6caffe26Tensor16raw_mutable_dataERKNS_8TypeMetaE) to reallocate the memory by calling `Context::New` and `meta_.ctor`.


### Other Memory Management Methods

In addition to constructors and data accessors, all other methods of `caffe2::Tensor` generally define the type and size of memory data accessors return:

   1. [`Resize`](http://yiwang.ngrok.io/codebrowser/caffe2/core/tensor.h.html#_ZN6caffe26Tensor6ResizeEDpT_),
   1. [`Reshape`](http://yiwang.ngrok.io/codebrowser/caffe2/core/tensor.h.html#_ZN6caffe26Tensor7ReshapeERKNSt3__16vectorIxNS1_9allocatorIxEEEE), a variant of `Resize` that requires the total number of elements doesn't change,
   1. [`ResizeLike`](http://yiwang.ngrok.io/codebrowser/caffe2/core/tensor.h.html#_ZN6caffe26Tensor10ResizeLikeERKNS_6TensorIT_EE), another variant of `Resize` that make the tensor the same size as another one,
   1. [`CopyFrom`](http://yiwang.ngrok.io/codebrowser/caffe2/core/tensor.h.html#_ZN6caffe26Tensor8CopyFromERKNS_6TensorIT_EEPT0_)
   1. [`Extend`](http://yiwang.ngrok.io/codebrowser/caffe2/core/tensor.h.html#_ZN6caffe26Tensor6ExtendExfPT_), which extends the out-most dimension,
   1. [`Shrink`](http://yiwang.ngrok.io/codebrowser/caffe2/core/tensor.h.html#_ZN6caffe26Tensor6ShrinkEx), which shrinks the out-most dimension,
   1. [`ShareData`](http://yiwang.ngrok.io/codebrowser/caffe2/core/tensor.h.html#_ZN6caffe26Tensor9ShareDataERKNS_6TensorIT_EE), which marks the current tensor shares another tensor's data, so `data()` and `mutable_data` returns (and mutate) another tensor's memory.


### Dims

The `vector<TIndex> Tensor::dims_` reads straightforward, and only some variants of [`SetDims`](http://yiwang.ngrok.io/codebrowser/caffe2/core/tensor.h.html#622) are required to construct it conveniently.


### Errors as Exceptions

Caffe2 throw exceptions when errors happen.  It has a set of macros sharing the prefix `CAFFE_ENFORCE`.   For example:

```cpp
    CAFFE_ENFORCE(
        outer_dim <= dims_[0],
        "New outer dimension must be smaller than current.");
```

where the definition of `CAFFE_ENFORCE` throws an exception:

```cpp
#define CAFFE_ENFORCE(condition, ...)                                         \
  do {                                                                        \
    if (!(condition)) {                                                       \
      throw ::caffe2::EnforceNotMet(                                          \
          __FILE__, __LINE__, #condition, ::caffe2::MakeString(__VA_ARGS__)); \
    }                                                                         \
  } while (false)
```

