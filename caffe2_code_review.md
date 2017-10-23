*** This is a draft! ***


### What is the difference between Run and RunAync

Page 9 shows SimpleNet::Run and SimpleNet::RunAsync.:q

Both calls ops in the topologically sorted order; whereas `SimpleNet::Run` calls `OperatorBase::Run(stream_id=0)`, and `SimpleNet::RunAsync` calls `OperatorBased::RunAsync(stream_id=0)`.

`OperatorBase::Run/RunAsync` are both virtual methods and are overloaded by `Operator`, which derives from `OperatorBase`.  Also, `Operator::Run/RunAsync` calls `Operator::RunOnDevice`, which is what users are supposed to overload when they define their own operators.  The difference  is that `Operator::Run` calls 

1. Context::SwitchToDeivce
1. Operator::Run
1. Context::FinishDevice

whereas `Operator::RunAsync` calls only the first two:

1. Context::SwitchToDeivce
1. Operator::Run

So `SimpleNet::Run` actually calls 



## From Python to C++

[`workspace.CreateNet`](https://github.com/caffe2/caffe2/blob/master/caffe2/python/workspace.py#L144) calls `C.create_net`:

```python
def CreateNet(net, overwrite=False, input_blobs=None):
    if input_blobs is None:
        input_blobs = []
    for input_blob in input_blobs:
        C.create_blob(input_blob)
    return CallWithExceptionIntercept(
        C.create_net, C.last_failed_op_uuid, StringifyProto(net), overwrite)
```

[`C.create_net`](https://github.com/caffe2/caffe2/blob/master/caffe2/python/pybind_state.cc#L781) calls `gWorkspace->CreateNet`:

```cpp
  m.def(
      "create_net",
      [](py::bytes net_def, bool overwrite) {
        caffe2::NetDef proto;
        CAFFE_ENFORCE(
            ParseProtobufFromLargeString(net_def, &proto),
            "Can't parse net proto: ",
            std::string(net_def));
        CAFFE_ENFORCE(
            gWorkspace->CreateNet(proto, overwrite),
            "Error creating net with proto: ",
            std::string(net_def));
        return true;
      },
      py::arg("net_def"),
      py::arg("overwrite") = kPyBindFalse);
```


where [`gWorkspace`](https://github.com/caffe2/caffe2/blob/master/caffe2/python/pybind_state.cc#L33) is of type `WorkSpace`.

[`Workspace::CreateNet`](https://github.com/caffe2/caffe2/blob/master/caffe2/core/workspace.cc#L147) [calls `caffe2::CreateNet`](https://github.com/caffe2/caffe2/blob/master/caffe2/core/workspace.cc#L167):

```cpp
  net_map_[net_def.name()] =
      unique_ptr<NetBase>(caffe2::CreateNet(net_def, this));
```


[`caffe2::CreateNet`](https://github.com/caffe2/caffe2/blob/master/caffe2/core/net.cc#L65) calls `SimpleNet`'s constructor via `make_unique`:

```cpp
unique_ptr<NetBase> CreateNet(const NetDef& net_def, Workspace* ws) {
  // In default, we will return a simple network that just runs all operators
  // sequentially.
  if (!net_def.has_type()) {
    return make_unique<SimpleNet>(net_def, ws);
  }
  return NetRegistry()->Create(net_def.type(), net_def, ws);
}
```

[`SimpleNet::SimpleNet`](https://github.com/caffe2/caffe2/blob/master/caffe2/core/net.cc#L74) call `SimpleNet::CreateOperator`:

```cpp
SimpleNet::SimpleNet(const NetDef& net_def, Workspace* ws)
    : NetBase(net_def, ws) {
  VLOG(1) << "Constructing SimpleNet " << net_def.name();
  bool net_def_has_device_option = net_def.has_device_option();
  // Initialize the operators
  for (const OperatorDef& operator_def : net_def.op()) {
    VLOG(1) << "Creating operator " << operator_def.name()
            << ":" << operator_def.type();
    if (!operator_def.has_device_option() && net_def_has_device_option) {
      // In the case that the operator def does not specify a device option but
      // the net def has a default option, we copy the device option over to the
      // operator def.
      OperatorDef temp_def(operator_def);
      temp_def.mutable_device_option()->CopyFrom(net_def.device_option());
      operators_.emplace_back(CreateOperator(temp_def, ws));
    } else {
      operators_.emplace_back(CreateOperator(operator_def, ws));
    }
  }
}
```

where [`CreateOperator`](https://github.com/caffe2/caffe2/blob/master/caffe2/core/operator.h#L562) is a global function

```cpp
unique_ptr<OperatorBase> CreateOperator(
    const OperatorDef& operator_def, Workspace* ws);
```

[whose implementation](https://github.com/caffe2/caffe2/blob/master/caffe2/core/operator.cc#L60) class [`OpertorBase`'s constructor](https://github.com/caffe2/caffe2/blob/master/caffe2/core/operator.cc#L17), which creates blobs according to `OperatorDef`:

```cpp
OperatorBase::OperatorBase(const OperatorDef& operator_def, Workspace* ws)
    : operator_ws_(ws),
      operator_def_(operator_def),
      arg_helper_(operator_def_) {
  ...

  for (const string& output_str : operator_def_.output()) {
    outputs_.push_back(CHECK_NOTNULL(ws->CreateBlob(output_str)));
  }
}
```

Please note that all input/output blobs are created by calling `Workspace::CreateBlob`, so the workspace would know all blobs.

Please be aware that the blob creation doesn't allocate tensor memory.  It is `Operator::Input/Output` who interpret blob's 

