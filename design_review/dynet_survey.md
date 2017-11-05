## dynet
dynet是一个非常轻量而又灵活的框架。它可以在书写神经网络计算公式时，构建出计算图。同时，forward/backward计算可以lazy执行，通常在计算图构建结束后执行。由于构建计算图的过程非常轻量，我们可以为每一个样本都构建出不同的计算图，实现动态神经网络。

下面将介绍dynet中的主要概念：

### Node

神经网络其实就是一个计算图，更确切的说，是一个有向无环图。而一个有向无环图本质上是由一定排列顺序的Node组成。Node上存储的是数据，Edge上描述了两个Node之间的联系，即数据之间的计算关系。

我们来看一下Node的定义，一个重要的成员变量是device，device实际上描述了在某个设备上的一块内存。

Node中还有一个指向ComputationGraph的指针，同时也包含了forward与backward的方法。

```
struct Device {
  int device_id;
  DeviceType type;
  MemAllocator* mem;
};


struct Node {
public:
  Dim dim;
  Device* device;
  
  virtual void forward_impl(const std::vector<const Tensor*>& xs,
                            Tensor& fx) const = 0;
                            
  virtual void backward_impl(const std::vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const = 0;
                             
  virtual void forward(const std::vector<const Tensor*>& xs,
                       Tensor& fx) const final;
                       
  virtual void backward(const std::vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const final;
  
private:
  ComputationGraph* cg_;
}；
```

### ComputationGraph

在这里，我们可以发现一个ComputationGraph是由一系列Node组成。


```
typedef unsigned VariableIndex;

struct ComputationGraph { 

  const Tensor& forward(const expr::Expression& last);
  void backward(const expr::Expression& last, bool full = false);
 
  std::vector<Node*> nodes;
  std::vector<VariableIndex> parameter_nodes;
  ExecutionEngine* ee;  
};
```


那么，如何构建一个ComputationGraph呢(即如何按照一定的顺序排列Node)？

dynet中通过调用以下几个方法向ComputationGraph中添加新的Node。


- add_input 加入一个输入数据的Node

```
VariableIndex ComputationGraph::add_input(const Dim& d, const vector<float>& pm) {
  VariableIndex new_node_index(nodes.size());
  nodes.push_back(new InputNode(d, pm));
  set_dim_for_new_node(new_node_index);
  return new_node_index;
}
```

- add_paramaters 加入一个参数的Node

```
VariableIndex ComputationGraph::add_parameters(Parameter p) {
  VariableIndex new_node_index(nodes.size());
  ParameterNode* new_node = new ParameterNode(p);
  nodes.push_back(new_node);
  parameter_nodes.push_back(new_node_index);
  set_dim_for_new_node(new_node_index);
  return new_node_index;
}
```

- add_lookup 

LookupNode的继承关系如下，用来表示把一个one-hot向量进行embedding操作之后的向量。

```
struct ParameterNodeBase : public Node
// represents a matrix/vector embedding of an item of a discrete set (1-hot coding)
struct LookupNode : public ParameterNodeBase
```

```
VariableIndex ComputationGraph::add_lookup(LookupParameter p, const unsigned* pindex) {
  VariableIndex new_node_index(nodes.size());
  LookupNode* new_node = new LookupNode(p, pindex);
  nodes.push_back(new_node);
  parameter_nodes.push_back(new_node_index);
  set_dim_for_new_node(new_node_index);
  return new_node_index;
}
```

- add_function 加入一个计算Node

```
template <class Function>
inline VariableIndex ComputationGraph::add_function(const std::initializer_list<VariableIndex>& arguments) {
  VariableIndex new_node_index(nodes.size());
  nodes.push_back(new Function(arguments));
  set_dim_for_new_node(new_node_index);
  return new_node_index;
}
```


### Expression

最终暴露给用户的是Expression，用户通过书写Expression公式，来构建ComputationGraph，向Graph中不断加入Node。

```
struct Expression {
  Computation* pg;
  VariableIndex i;
}
```

Expression的书写过程可以参考下面的例子：

```
Model model;
Parameter pW = model.add_parameter({1});

SimpleSGDTrainer trainer(model, 1.0);

ComputationGraph cg;
Expression W = parameter(cg, W);

for (unsigned i = 0; i < xs.size(); i++) {
  Expression pred = W * sx[i];
  Expression loss = square(pred - ys[i]);
  cg.forward(loss);
  cg.backward(loss);
  trainer.update();
}
```

用户可以把输入数据，parameters转换为Expression。同时，也可以书写Expression之间的计算。在书写Expression的过程中，会调用```add_input, add_parameters, add_lookup, add_function```等方法，Node会不断的被加入到cg中。

dynet中给出了很多细粒度的Node实现，比如：

```
struct Square : public Node
struct Cube : public Node
struct Exp : public Node
```

这些Node中的forward，backward的计算最后会调用Eigen库执行。

### Model

Model存放着所有的parameters，并且可以被Trainer进行更新。注意，这里parameters中只包含value与gradient，而一些优化方法中需要的动量等信息，则由对应的Trainer负责分配。

```
struct ParameterStorage : public ParameterStorageBase {
public:
  Tensor values;
  Tensor g;
};

class Model {
private:
  std::vector<ParameterStorageBase*> all_params;
  std::vector<ParameterStorage*> params;
  std::vector<LookupParameterStorage*> lookup_params;
};
```


### ExecutionEngine

当一个Expression书写完毕之后，整个ComputationGraph也构建完毕，我们需要执行整个ComputationGraph，会调用forward与backward方法。在这里实际上是调用ExecutionEngine的成员方法。

```
class ExecutionEngine {
public:
  virtual const Tensor& forward(VariableIndex i) = 0;
  virtual void backward(VariableIndex i, bool full = false) = 0;

private:
  const ComputationGraph& cg;
  VariableIndex backward_computed;
}
```

ExecutinEngine会派生出SimpleExecutionEngine。在这里，最后会在```incremental_forward```方法中完成所有Node的forward过程。

```
const Tensor& SimpleExecutionEngine::incremental_forward(VariableIndex i) {
  ...
  for (; num_nodes_evaluated <= i; ++num_nodes_evaluated) {
    ...
    const Node* node = cg.nodes[num_nodes_evaluated];
    ...
    node->forward(xs, nfxs[num_nodes_evaluated]);
    ...
  }
}
```

而反向过程如下所示：
```
void SimpleExecutionEngine::backward(VariableIndex from_where, bool full) {
  ...
  for (int i = num_nodes - 1; i >= 0; --i) {
    const Node* node = cg.nodes[i];
    ...
    for (VariableIndex arg : node->args) {
      ...
      if (needs_derivative[arg]) {
        node->backward(xs, nfxs[i], ndEdfs[i], ai, ndEdfs[arg]);
      }
      ++ai;
    }
  }
  
  // accumulate gradients into parameters
  // this is simpler than you might find in some other frameworks
  // since we assume parameters come into the graph as a "function"
  // that returns the current value of the parameters
  for (VariableIndex i : cg.parameter_nodes) {
    static_cast<ParameterNodeBase*>(cg.nodes[i])->accumulate_grad(ndEdfs[i]);
  }
  ...
}
```

同时在反向计算完毕之后，gradients也会被更新到parameters(parameter都被放置在Model中，实际上Model包含两个vector，一个存储value，另一个存储gradient，这里是把计算得到的gradient值写入Model中对应的gradient的vector中)。


### Trainer

实际上是Optimizer的功能，对参数按照一定的计算公式进行更新。

Trainer会派生出如下类型：

- SimpleSGDTrainer
- CyclicalSGDTrainer
- MomentumSGDTrainer
- AdagradTrainer
- AdadeltaTrainer
- RMSPropTrainer
- AdamTrainer


分别有自己的更新规则，以AdagradTrainer为例：

```
// Perform update of ts[0]=parameters, ts[1]=gradients, ts[2]=stddev
template <class MyDevice>
void AdagradTrainer::update_rule_dev(const MyDevice & dev, real scale, real gscale, const std::vector<Tensor*> & ts) {
  ts[1]->tvec().device(*dev.edevice) = ts[1]->tvec() * (scale * gscale);
  ts[2]->tvec().device(*dev.edevice) += ts[1]->tvec().square();
  ts[0]->tvec().device(*dev.edevice) += ts[1]->tvec() / (ts[2]->tvec() + epsilon).sqrt() * (-eta / model->weight_decay.current_weight_decay());
}
```

### 总结

dynet目前forward与backward计算过程都在一个Node内，因此，目前整个计算图的执行过程仍然是顺序进行的，目前由SimpleExecutionEngine负责执行。我们可以把把backward过程显式的用单独的Node进行表示，这样可以对DAG中Node之间的依赖关系进行分析，实现相应的DAGExecutionEngine，从而可以提高并行度。

