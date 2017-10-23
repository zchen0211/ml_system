
## Context


`Context`是与特定设备相关的操作的封装，其中包含了在具体设备上运行Operator所必须的各种内容。它作为Tensor和Operator的模板参数，用于特化出不同设备上的Tensor和Operator。

```c++
template <class Context> class Tensor {};

template <class Context>
class Operartor : public OperarorBase {};
```
根据具体设备的不同，Context可以有各种各样的类型，例如CPUContext，CUDAContext等。所有的Context都必须实现以下几个函数：

- `static void* New(size_t nbytes)`设备上的内存申请
- `static void* Delete(void* data)`设备上的内存释放
- `void SwitchToDevice()`实现切换设备的方法
- `bool FinishDeviceComputation()`Operator在设备上运行结束后的收尾工作
- `template <class SrcContext, class DstContext> void CopyBytes(...)`实现不同设备之间相互拷贝的方法
- `template <typename T, class SrcContext, class DstContext> void Copy(...)`对`void CopyBytes(...)`的封装

当然，在这些必须的函数基础上，Context可以根据自己的实际情况包含一些自己独有的函数，例如：
- 负责特定设备上的资源管理，比如CUDAContext包含的stream，cublasHandle等
- 负责随机数的生成

在Tensor和Operator中，包含一个Context类型的成员变量`context_`。当Tensor和Operator需要执行与设备相关的操作时，就调用`context_`中的对应函数来实现。

## Tensor 

对特定设备上一段连续内存的View，包含一个指向内存块的std::shared_ptr<void>指针

```c++
typedef int64_t TIndex;
template <class Context>
class Tensor {
protected:
  vector<TIndex> dims_;
  TypeMeta meta_;
  std::shared_ptr<void> data_;
};
```

`data_`指向一块连续的内存，存放了Tensor的具体数据，`meta_`中存储了该块内存的数据类型信息。

Tensor并未包含计算的部分，在Caffe2中，部分CPU上的计算调用Eigen库完成，其他则采用在Operator中直接手写循环的方式实现。

更详细的内容可以参考 [[Caffe2::Tensor]]

## TypeMeta
在Tensor的实现中，模板参数只有Context，并没有任何与数据类型相关的参数。为了描述Tensor中保存的数据的类型，Tensor中包含一个TypeMeta类型的成员变量`meta_`。

TypeMeta主要包含以下内容：
- `CaffeTypeId id_`类型T的唯一id，不同类型的id一定不同
- `size_t itemsize_`类型T的`sizeof`结果
- `PlacementNew ctor_`函数指针，用于初始化T类型数组中的每一个对象，数组长度作为参数传入
- `TypedCopy copy_`函数指针，用于拷贝T类型的数组
- `TypedDestructor dtor_`函数指针，用于销毁T类型数组中的每一个对象。数组长度作为参数传入

如果类型T的定义已经确定，那么以上各项其实也都已经唯一确定。因此TypeMeta提供了Make函数模板来自动生成类型T对应的TypeMeta：
```c++
template <
    typename T,
    typename std::enable_if<
        !std::is_fundamental<T>::value &&
        std::is_copy_assignable<T>::value>::type* = nullptr>
static TypeMeta Make() {
  return TypeMeta(Id<T>(), ItemSize<T>(), _Ctor<T>, _Copy<T>, _Dtor<T>);
}
```
其中`_Ctor<T>`、`_Copy<T>`、`_Dtor<T>`都是预先定义好的函数模板，例如数组对象初始化函数`_Ctor<T>`的定义：
```c++
template <typename T>
static void _Ctor(void* ptr, size_t n) {
  T* typed_ptr = static_cast<T*>(ptr);
  for (int i = 0; i < n; ++i) {
    new (typed_ptr + i) T;
  }
}
```

需要注意：虽然TypeMeta中包含了对T类型的数组中每个对象的析构操作，但是并不包含对数组本身的内存释放，数组本身的释放由Context中的Detele函数完成。因此完全释放一个Tensor的内存应该先调用`TypeMeta::Dtor<T>()`再调用`Context::Delete()`。

## Blob
Blob是一个通用的存放一个有类型指针的容器，通常情况下存放的是Tensor。 Blob利用成员变量`void* pointer_`指向被存放的对象。

```c++
class Blob {
public:
  Blob() : meta_(), pointer_(nullptr) {}
  ~Blob() { Reset(); }
private:
  TypeMeta meta_;
  void* pointer_ = nullptr;  
};
```
Operator的输入输出均为Blob。在需要执行Tensor计算时，首先从Blob中取出pointer_指针并转换成Tensor<Context>类型，然后调用Tensor提供的data函数获取数据区域的裸指针，最后手写循环或者调用Eigen库进行计算。

## Operator

Operator在特定设备上执行，所有的派生类必须实现RunOnDevice的方法，用于描述在特定设备上执行的计算。


```c++
class OperatorBase {
private:
  OperatorDef operator_def_;
  vector<const Blob*> inputs_;
  vector<Blob*> outputs_;
}


template <class Context>
class Operator : public OperatorBase {
public:
  virtual bool RunOnDevice() = 0;
protected:
  Context context_;  
};
```

描述神经网络forward与backward计算的Operator是分开的，Caffe2实现了一套注册机制，用于方便的增加新的Operator。

在Caffe2中所有计算都使用Operator进行表示，包括optimizer，多GPU通信，多机通信，数据载入等。通常情况下Operator可以分为如下几类：

- 数据加载与存储：DBExistsOp/LoadOp/SaveOp/CheckpointOp，操作数据库DB，完成数据的读取与保存
- 初始化数据: 对数据进行初始化填充操作，实现了FillerOp基类，派生出UniformFillOp/UniqueUniformFillOp/ConstantFillOp/GaussianFillOp等
- forward/backward计算：包括ReluOp/ReluGradientOp/FullyConnectedOp/FullyConnectedGradientOp等，所有的backward都有对应的Op，其中一部分可以通过注册机制得到，一部分需要手写得到
- 单机多卡/多机多卡梯度聚合：包括AllreduceOp/BroadcastOp等，统一在contrib/gloo目录下
- 参数更新：AdamOp/AdagradOp/MomentumSGDOp等

下面举例进行说明：

```c++
template <typename T, class Context>
class AdagradOp final : public Operator<Context> {};


class NCCLAllreduceOp final : public Operator<CUDAContext> {};
class NCCLBroadcastOp final : public Operator<CUDAContext> {};


template <class Context>
class AllreduceOp final : public Operator<Context> {};

template <class Context>
class BroadcastOp final : public Operator<Context> {}; 


template <class Context>
class CreateDBOp final : Operator<Context> {
public:
  bool RunOnDevice() final {
    OperatorBase::Output<db::DBReader>(0)->Open(
        db_type_, db_name_, num_shards_, shard_id_);
    return true;
  }
};
```

## Registry和OpSchemaRegistry

Caffe2中对Operator的注册分为三个部分：

- 对Op构造函数的注册  
- 对Op的基本特性的注册，基本特性包括输入输出的参数个数等各项配置  
- 对Op所对应的backword Op的注册。在Caffe2中，forword和backword操作在不同的Op中实现，因此需要对他们之间的对应关系进行注册。

构造函数和backword Op注册在Registry模板类中。其模板参数和主要成员变量有：

```c++
template <class SrcType, class ObjectType, class... Args>
class Registry {
  typedef std::function<std::unique_ptr<ObjectType> (Args ...)> Creator;

  CaffeMap<SrcType, Creator> registry_;
  CaffeMap<SrcType, string> help_message_;
  std::mutex register_mutex_;
};
```
SrcType一般是std::string字符串，ObjectType则为某一种class。Args表示构造这种class可能需要的各项参数的类型。  

注册信息保存在registry_中，这本质上是一个map。SrcType一般表示被注册的Op的名称，Creator则随注册场景而变化。

对Op基本特性的注册则在OpSchemaRegistry类中实现。与Registry类似，其核心也是一个map：

```c++
static CaffeMap<string, OpSchema>& map();
```

OpSchema类专门用于存放Operator的各项基本特性。


## Net

Net中包含了一系列Operator。Caffe2中基于NetBase，派生出了SimpleNet。


```c++
class NetBase {
public:
  virtual bool Run() = 0;
};


class SimpleNet : public NetBase {
protected:
  vector<unique_ptr<OperatorBase>> operarors_;
};

```

SimpleNet会依次执行所包含的Operator。同时，为了进一步的优化神经网络的执行过程，Caffe2也提供了DAGNetBase。DAGNetBase要较为复杂，同时需要实现自己的执行引擎。

```c++
namespace internal {
struct OperatorNode {
  unique_ptr<OperatorBase> operator_;
  vector<int> children_;
  vector<int> parents_;
  std::atomic<int> runtime_parent_count_;
  bool is_chain_statr_ = false;
};

struct OpGraphNode {
  vector<int> children_;
  vector<int> parents_;
  int visited_inputs = 0;
  int num_orig_parents_;
};
}

class DAGNetBase : public NetBase {

protected:
  virtual bool RunAt(const std::vector<int>& chain) = 0;
  
  std::vector<internal::OperatorNode> operator_nodes_;
  ExecutionChains execution_chains_;
  std::vector<std::thread> workers_;
};


class DAGNet : public DAGNetBase {};

class AsyncDAGNet : public DAGNetBase {
protected:
  std::vector<int32_t> eventRecorded_;
  std::vector<std::unique_ptr<internal::Event>> events_;  
};
```

## Workspace

Workspace中包含了一切运行时创建的对象，包括所有的Blob，以及Net等。

```c++
class Workspace {
private:
  BlobMap blob_map_;
  NetMap net_map_;
};
```

## Python绑定

- C++端注册的Operator都会被引入到Python端
- 计算图的构建是在Python端完成的，包括forward，backward，以及update，checkpoint过程，都是组合注册得到的Op
- 用户在Python端定义神经网络的配置，通常只需要书写forward的过程；而backward的过程与参数update的过程，caffe2在Python端封装了简易接口，可以快速的搭建完整的计算图
- 计算图会序列化为protobuf文件，然后传入C++端，开始执行网络的计算过程

### 添加梯度计算的相关Op
用户只需要书写前向的计算过程，反向的计算过程是由python前端构建的，需要显式的调用AddGradientOperators方法，返回一个存储所有gradient的Blob的一个map。相关调用过程如下：


```
// caffe2/python/core.py
def _GenerateGradientsForForwardOp

// 这个函数得到计算给定Blob的梯度的过程，会调用_GenerateGradientsForForwardOp方法
def GetBackwardPass

// 这个函数会调用GetBackwardPass
def AddGradientOperators

```

### 添加参数更新的相关Op

这个操作是由用户来做的

当然，caffe2也提供了ModelHelper来去简化过程，ModelHelper中会记录所有的params的Blob，以及params_to_grad的Blob，用户只需要书写对应的更新过程就可以了。ModelHelper实际上也是一个辅助存放参数Blob的集合


### 单机多卡/多机多卡

Parallelize_GPU来封装单机多卡/多机多卡的model，主要的调用过程如下：

```
input_builder_fun
forward_pass_builder_fun
_ValidateParams
_AddGradientOperators
_AllReduceGradients
```

## 总结
Caffe2的一些有待考虑的设计如下：

- 不支持multi-thread训练
- 通过NCCL库支持multi-GPU训练；在C++端封装NCCL相关的allreduce，gather等op；然后在Python端实现parallize_model，调用NCCL的相关op
- 与Python强绑定；使用pybind11库来直接暴露C++的API；没有实现C-API，无法增加多语言的绑定

Caffe2的优点：

- CPU代码和GPU代码组织清晰，编译简洁
- 所有的操作都抽象为Operator，接口统一
- 提供了顺序执行的SimpleNet，以及可以并行执行的DAGNet

