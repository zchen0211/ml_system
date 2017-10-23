### [设计思想](http://mxnet.io/architecture/program_model.html)

这篇文章详细比较了Symbolic和Imperative两种神经网络编程风格的优缺点：
#### 主要结论：
1. Symbolic风格效率更高。
原因在于构建完计算图之后可以做优化，比如inplace内存管理，operator合并等。而动态网络，因为中间结果可能随时会被用到，所以所有中间状态都需要保存下来。

2. [反向传播和autodiff如何做](http://mxnet.io/architecture/program_model.html#case-study-backprop-and-autodiff)


### Mxnet主要数据结构：
1. Symbol
1. Graph
1. Node
1. NodeEntry


### [Symbol](http://www.superjom.xyz/mxnetcode/codebrowser/nnvm/include/nnvm/symbolic.h.html#nnvm::Symbol)

Symbol对外提供的操作自己的接口，基本都是在操作[`vector<NodeEntry> outputs`](http://www.superjom.xyz/mxnetcode/codebrowser/nnvm/include/nnvm/symbolic.h.html#nnvm::Symbol::outputs)，`Graph`也是操作同样的数据结构，所以他们通过这个outputs进行转换，后面会分析。
```cpp
  /*! \brief output entries contained in the symbol */
  std::vector<NodeEntry> outputs;
```

例如下面是创建一个VariableNode：
```cpp
Symbol Symbol::CreateVariable(const std::string& name) {
  Symbol s;
  s.outputs.emplace_back(NodeEntry{CreateVariableNode(name), 0, 0});
  return s;
}
```

### [Graph](http://www.superjom.xyz/mxnetcode/codebrowser/nnvm/include/nnvm/graph.h.html#nnvm::Graph)

Graph是内部逻辑表示，用户构建的Symbol会先转换成Graph，然后会有一些优化函数(optimize pass)对Graph进行优化，做一些InPlace, operator fusion之类的操作。对外最主要的成员是一个由NodeEntry组成的Vector [output](http://www.superjom.xyz/mxnetcode/codebrowser/nnvm/include/nnvm/graph.h.html#nnvm::Graph::outputs)
内部为了优化方便，还有一个[IndexedGraph](http://www.superjom.xyz/mxnetcode/codebrowser/nnvm/include/nnvm/graph.h.html#nnvm::Graph::indexed_graph_)


### Symbol和Graph的关系
Symbol对外提供接口(包括python)，帮助用户构建计算图，Graph内部使用，构建完的Symbol需要先转换成Graph，然后执行运算。
`Symbol`和`Graph`之间的构建关系，从下面可以看出，他们之间通过`outputs`进行沟通，而`outputs`都是std::vector<NodeEntry>类型的。

例如：
```python
def test_infer_shape():
    x = sym.Variable('x', shape=(4, 2))
    y = sym.add(x, x, name='add1')
    y = sym.reshape(y, target=(2, 4), name="reshape1")
    g = graph.create(y)
    g._set_json_attr("shape_attr_key", "shape")
    g = g.apply('InferShape')
    jgraph = json.loads(g.apply('SaveJSON').json_attr('json'))
    jnodes = jgraph['nodes']
    jnode_row_ptr = jgraph['node_row_ptr']
    nindex = {n['name']: i for i, n in enumerate(jnodes)}
    assert g.json_attr('shape')[jnode_row_ptr[nindex["reshape1"]]] == [2, 4]
    assert g.json_attr('shape')[jnode_row_ptr[nindex["add1"]]] == [4, 2]
```

GraphExecutor初始化：
```cpp
  nnvm::Graph g = InitGraph(symbol, default_ctx, ctx_map, in_arg_ctxes,
                            arg_grad_ctxes, aux_state_ctxes, grad_req_types);
```

可见都是先构建Symbol，然后将基于Symbol Create一个Graph出来，再做后边的事情。

```cpp
int NNGraphCreate(SymbolHandle symbol, GraphHandle *graph) {
  Graph* g = new Graph();
  API_BEGIN();
  g->outputs = static_cast<Symbol*>(symbol)->outputs;
  *graph = g;
  API_END_HANDLE_ERROR(delete g);
}
```

### [Node](http://www.superjom.xyz/mxnetcode/codebrowser/nnvm/include/nnvm/node.h.html#nnvm::Node)

graph由Node组成：
```cpp
 * \brief Node represents an operation in a computation graph.
```

Node中包含一个成员变量NodeAttrs,可以从NodeAttrs中获取到Op，而Op在注册的时候注册了一个AttriBute比如上边的`AddKernel`。于是和计算绑定在一起了。可以通过`string` ==> `OpKernel<gpu>` 找到这个对应的kernel。

Variable也是Node，特点是他的op为nullptr。
```cpp
inline bool Node::is_variable() const {
  return this->op() == nullptr;
}
```

### [NodeEntry](http://www.superjom.xyz/mxnetcode/codebrowser/nnvm/include/nnvm/node.h.html#nnvm::NodeEntry)
an entry that represents output data from a node

```cpp
/*! \brief an entry that represents output data from a node */
struct NodeEntry {
  /*! \brief the source node of this data */
  NodePtr node;
  /*! \brief index of output from the source. */
  uint32_t index;
  /*!
   * \brief version of input Variable.
   *  This field can only be nonzero when this->node is a Variable node.
   *  version is increased by one each time a Variable get composed to a mutation Op.
   *  This information can be helpful to decide order of operations when sequence of mutation happens.
   */
  uint32_t version;
};
```


### register operator.
kernel也是作为一种通用attr注册进去的，

attr是一个string到any的映射，装了很多种东西：
```cpp
std::unordered_map<std::string, std::unique_ptr<any> > attr;
```

```cpp
// registeration of oeprators
// NOTE that the attr function can register any
// additional attributes to the operator
NNVM_REGISTER_OP(add)
.describe("add two inputs together")
.set_num_inputs(2)
.set_attr<OpKernel>("OpKernel<gpu>", AddKernel)
.include("ElementwiseOpAttr");
```

set_attr实际上调用了UpdateAttrMap：

有一个全局的OpManager，负责管理key：attr对，attr的类型为
```cpp
std::unordered_map<std::string, std::unique_ptr<any> > attr;
```

```cpp
// update attribute map
void Op::UpdateAttrMap(const std::string& key,
                       std::function<void(any*)> updater) {
  OpManager* mgr = OpManager::Global();
  std::lock_guard<std::recursive_mutex>(mgr->mutex);
  std::unique_ptr<any>& value = mgr->attr[key];
  if (value.get() == nullptr) value.reset(new any());
  if (updater != nullptr) updater(value.get());
}
```

