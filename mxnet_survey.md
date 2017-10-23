<div id="table-of-contents">
<h2>Table of Contents</h2>
<div id="text-table-of-contents">
<ul>
<li><a href="#org6cf7533">1. 总结</a></li>
<li><a href="#org9068b8d">2. API 及功能</a>
<ul>
<li><a href="#orga634648">2.1. engine 接口</a></li>
<li><a href="#orgd1c7021">2.2. <code>threaded_engine</code>  实现</a>
<ul>
<li><a href="#org94c1113">2.2.1. <code>Var</code></a></li>
<li><a href="#org35c329f">2.2.2. AppendReadDependency</a></li>
<li><a href="#org6cd9303">2.2.3. AppendWriteDependency</a></li>
<li><a href="#orgc165026">2.2.4. CompleteReadDependency</a></li>
<li><a href="#org2c14aeb">2.2.5. CompleteWriteDependency</a></li>
</ul>
</li>
<li><a href="#orgb62c3f4">2.3. Engine 总接口</a></li>
<li><a href="#org40885ae">2.4. ThreadedEnginePooled</a></li>
<li><a href="#orgc0e354d">2.5. ThreadedEnginePerDevice</a></li>
</ul>
</li>
<li><a href="#orgcdbea0b">3. 参考文献</a></li>
</ul>
</div>
</div>


<a id="org6cf7533"></a>

# 总结

`mxnet::engine` 主要包括如下实现：

-   function 并行执行过程中的参数依赖问题
-   精确到 device 的多线程调度控制

除了具体实现之外，可以借鉴的设计思想：

-   每个 device 分配自己的任务队列和线程池，function 分配到具体 device 执行
    -   便于更可控的性能调度
-   普通任务通过设置 `device id` 分配到具体的 device 上执行
-   设立 high priority 专用线程池，不区分 device，所有 device 资源优先执行高优先任务
-   CPU/GPU 间的拷贝操作单独拆开，用 IO 专用线程池专门负责，保证与计算任务间并发
    -   每个 device 默认只设 1 个线程负责 IO，因为同一个 device 的 IO 无法支持高效并发
-   在实现一个复杂模块前，用一个 naive 的实现验证接口和基本功能
-   模块设立 `profiler` 来追踪执行及性能情况，方便人工分析


<a id="org9068b8d"></a>

# API 及功能

这里完全参考官方文档[1]里的内容

`mxnet::engine` 的功能是，按照依赖关系执行多个 function，其执行有如下原则

-   有依赖关系的 function 必须依次执行
-   无依赖关系的 function 间并行执行

执行的主要 API 如下：

```c++
virtual void PushSync(Fn exec_fun, Context exec_ctx,
                      std::vector<VarHandle> const& const_vars,
                      std::vector<VarHandle> const& mutate_vars) = 0;
```

-   `threaded_engine.h/.cc` , 基于线程池的 engine
    -   `thread_engine_pooled` 所有 device 共用一个任务队列的实现
    -   `threaded_engine_perdevice.h/.cc`  每个 device 单独分配任务队列的实现，针对 CPU/GPU 性能方面的考虑
-   `thread_pool.h` , 一个简单线程池的实现

代码的逻辑是

-   `engine.h`  做对外接口，其中提供一个单例 `static Engine* Engine::Get()` 来获取底层具体的 engine 实例；
-   `naive_engine.cc` , `threaded_engine_pooled.cc` , `threaded_engine_perdevice.cc`  三个文件实现了三种 engine；
-   `profiler.cc` 实现了 `class Profiler` 来追踪 `mxnet::engine` 运行中的信息，方便性能调优和 debug。


<a id="orga634648"></a>

## engine 接口

```c++
/*!
 * \brief Dependency engine that schedules operations.
*/
class MXNET_API Engine {
 public:
  /*! \brief callback on complete*/
  typedef engine::CallbackOnComplete CallbackOnComplete;
  /*! \brief Synchronous operation to pass to engine. */
  typedef std::function<void(RunContext)> SyncFn;
  /*! \brief Asynchronous operation to pass to engine. */
  typedef std::function<void(RunContext, CallbackOnComplete)> AsyncFn;
  /*! \brief Variable pointer */
  typedef engine::VarHandle VarHandle;
  /*! \brief Operator pointer */
  typedef engine::OprHandle OprHandle;
  /*!
   * \brief Notify the engine about a shutdown,
   *  This can help engine to print less messages into display.
   *
   *  User do not have to call this function.
   * \return 0 when success, -1 when failure happens.
   */
  virtual void NotifyShutdown() = 0;
  /*!
   * \brief Allocate a new variable, the variable can then
   *        be used to schedule the operation concurrently via dependency
   *        patterns.
   * \return The new variable allocated.
   */
  virtual VarHandle NewVariable() = 0;
  /*!
   * \brief Create a new operator. The returned operator could be saved
   *        externally so that it could be resued for scheduling.
   * \param fn The execution function.
   * \param const_vars The variables that current operation will use but not
   *                   mutate.
   * \param mutable_vars The variables that current operation will mutate.
   * \param prop Property of the function.
   * \param opr_name The operator name.
   * \return The new operator allocated.
   */
  virtual OprHandle NewOperator(AsyncFn fn,
                                std::vector<VarHandle> const& const_vars,
                                std::vector<VarHandle> const& mutable_vars,
                                FnProperty prop = FnProperty::kNormal,
                                const char* opr_name = nullptr) = 0;
  /*!
   * \brief Delete the given operator.
   * \param op The operator to delete.
   *
   * The delete will not happen immediately, but will wait until all the
   * operations using this operator are completed.
   */
  virtual void DeleteOperator(OprHandle op) = 0;
  /*!
   * \brief Push an operator to the engine.
   * \param op The operator to push.
   * \param exec_ctx Execution context.
   * \param priority Priority of the action, as hint to the engine.
   * \param profiling The variable indicate whether to profile this operator.
   */
  virtual void Push(OprHandle op, Context exec_ctx, int priority = 0, bool profiling = false) = 0;
  /*!
   * \brief Push an asynchronous operation to the engine.
   * \param exec_fun Execution function, this function takes a parameter
   *                 on_complete that must be called when the execution
   *                 completes.
   * \param exec_ctx Execution context.
   * \param const_vars The variables that current operation will use but not
   *                   mutate.
   * \param mutable_vars The variables that current operation will mutate.
   * \param prop Property of the function.
   * \param priority Priority of the action, as hint to the engine.
   * \param opr_name The operator name.
   */
  virtual void PushAsync(AsyncFn exec_fun, Context exec_ctx,
                         std::vector<VarHandle> const& const_vars,
                         std::vector<VarHandle> const& mutable_vars,
                         FnProperty prop = FnProperty::kNormal,
                         int priority = 0,
                         const char* opr_name = nullptr) = 0;
  /*!
   * \brief Schedule the deletion of a variable.
   *
   * The delete will not happen immediately, but will wait until all the
   * operations depending on var are completed.
   *
   * \param delete_fn A function that will be called after the variable is
   *                   deleted.
   * \param exec_ctx Execution context.
   * \param var The variable to be deleted.
   */
  virtual void DeleteVariable(SyncFn delete_fn,
                              Context exec_ctx,
                              VarHandle var) = 0;
  /*!
   * \brief Wait for a variable.
   * \param var The variable we should wait for. This function returns when the
   *            variable is ready.
   */
  virtual void WaitForVar(VarHandle var) = 0;
  /*!
   * \brief Wait until all the activity of engine finishes.
   */
  virtual void WaitForAll() = 0;
  /*!\brief virtual destructor */
  virtual ~Engine() noexcept(false) {}
  /*!
   * \return Engine singleton.
   */
  static Engine* Get();
  /*!
   * \brief Get shared pointer reference to engine singleton.
   *  Most user should not call this function.
   *  This function is called by another singleton X who requires
   *  engine to be destructed after X.
   *
   * \return A shared pointer to Engine singleton.
   */
  static std::shared_ptr<Engine> _GetSharedRef();
  /*!
   * \brief Push an synchronous operation to the engine.
   * \param exec_fn Execution function that executes the operation.
   * \param exec_ctx Execution context.
   * \param const_vars The variables that current operation will use but not
   *                   mutate.
   * \param mutable_vars The variables that current operation will mutate.
   * \param prop Property of the function.
   * \param priority Priority of the action, as hint to the engine.
   * \param opr_name The operator name.
   * \tparam SyncFn the synchronous function to be pushed.
   */
  inline void PushSync(SyncFn exec_fn, Context exec_ctx,
                       std::vector<VarHandle> const& const_vars,
                       std::vector<VarHandle> const& mutable_vars,
                       FnProperty prop = FnProperty::kNormal,
                       int priority = 0,
                       const char* opr_name = nullptr) {
    this->PushAsync([exec_fn](RunContext ctx, CallbackOnComplete on_complete) {
        exec_fn(ctx);
        on_complete();
      }, exec_ctx, const_vars, mutable_vars, prop, priority, opr_name);
  }

  /*!
   * \brief factory function to create OnComplete callback.
   * \param callback th static callback function.
   * \param param the paramter passed to callback.
   */
  inline CallbackOnComplete CreateCallback(
      void (*callback)(Engine *, void *), void *param) {
    CallbackOnComplete ret;
    ret.callback_ = callback;
    ret.engine_ = this;
    ret.param_ = param;
    return ret;
  }
};  // class Engine
```

<a id="orgd1c7021"></a>

## `threaded_engine`  实现

在 `thread_engine.h` 中包括了实现中的一些概念，比如


<a id="org94c1113"></a>

### `Var`

`engine.h` 中定义的 `Var` 用来管理依赖某个 variable 后多个 function 的先后操作关系。

```c++
class ThreadedVar final : public Var,
                          public common::ObjectPoolAllocatable<ThreadedVar>
```

其中， `ThreadedVar` 是一个 FIFO 链表 queue，链表中的每个节点是

```c++
/*!
 * \brief VersionedVarBlock that corresponding to a variable version.
 *  This is a basic unit of LinkedList in the ThreadedVar.
 */
struct VersionedVarBlock
  : public common::ObjectPoolAllocatable<VersionedVarBlock> {
  /*! \brief next block in the LinkedList */
  VersionedVarBlock* next{nullptr};
  /*! \brief the operation this block triggers */
  OprBlock* trigger{nullptr};
  /*! \brief whether this operation is a write(mutate) operation. */
  bool write{false};
  /*! \brief define possible debug information */
  DEFINE_ENGINE_DEBUG_INFO(VersionedVarBlock);
};  // struct VersionedVarBlock
```

每个 `VersionedVarBlock` 表示一个依赖该 `Var` 的 function，
`ThreadedVar` 用一个链表表示 FIFO 队列，来管理所有的 `VersionedVarBlock` ，即依赖的 function。

链表的结构通过如下 member variable 表示：

```c++
VersionedVarBlock* pending_write_{nullptr};
```

-   `pending_write_` 指向链表队列中最前面（最旧）的 请求 Write 操作的 `VersionedVarBlock`
-   `pedding_write_` 其实是链表的 HEAD，因为在 所有 Write 操作前的 Read 操作会直接调度执行， 并不会进入链表（参照 <a href="#AppendReadDependency">AppendReadDependency</a>）

```c++
VersionedVarBlock* head_{nullptr};
```

`head_` 指向链表队列末尾的位置（名字太有迷惑性了。。）， 当需要添加新的元素时只需要

```c++
head_->next = new_var_block;
head_->trigger = opr_block;
head_ = new_var_block;
```

`ThreadedVar` 调度 function 依赖关系的过程就是对 `VersionedVarBlock` 的链表的维护过程，
具体的管理过程包括如下 4 个 API：

-   `void ThreadedVar::AppendReadDependency(OprBlock* opr_block);`
    -   添加 Read 依赖
-   `void ThreadedVar::AppendWriteDependency(OprBlock* opr_block);`
    -   添加 Write 依赖
-   `void ThreadedVar::CompleteReadDependency(Dispatcher dispatcher)`
    -   Read 依赖完成
-   `bool ThreadedVar::CompleteWriteDependency(Dispatcher dispatcher)`
    -   Write 依赖完成

<a id="AppendReadDependency"></a>


<a id="org35c329f"></a>

### AppendReadDependency

添加 Read 依赖的主要逻辑是

-   如果链表队列没有 padding 的 Write 操作依赖（ `pending_write_ = nullptr` ）
    -   则根据<a href="#rule0">规则</a> 该 function 的 Read 依赖直接满足，通过 `opr_block->decr_wait()`
    -   该 `opr_block` 无需加入到链表队列中
-   否则
    -   乖乖 append 到队列末尾

```c++
inline void ThreadedVar::AppendReadDependency(OprBlock* opr_block) {
  std::lock_guard<std::mutex> lock{m_};
  if (pending_write_ == nullptr) {
    // invariant: is_ready_to_read()
    CHECK_GE(num_pending_reads_, 0);
    // STATE CHANGE
    ++num_pending_reads_;
    // decrease wait counter
    opr_block->decr_wait();
  } else {
    auto&& new_var_block = VersionedVarBlock::New();
    assert(head_->next == nullptr);
    assert(head_->trigger == nullptr);
    assert(head_->write == false);
    // append things to next.
    head_->next = new_var_block;
    head_->trigger = opr_block;
    head_ = new_var_block;
  }
}
```

其中， `num_pedding_reads_` 只是一个 state，用于表示是否还有 Read 依赖，在判定能否删除该 `Var` 会用到。


<a id="org6cd9303"></a>

### AppendWriteDependency

添加 Write 依赖，由于 必然会产生 <a href="#rule0">规则</a> 中描述的 Read 和 Write 的问题，
因此必须要追加到队列末尾按顺序执行。

```c++
inline void ThreadedVar::AppendWriteDependency(OprBlock* opr_block) {
  auto&& new_var_block = VersionedVarBlock::New();
  std::lock_guard<std::mutex> lock{m_};
  // invariant.
  assert(head_->next == nullptr);
  assert(head_->trigger == nullptr);
  assert(head_->write == false);
  // attach to head.
  head_->next = new_var_block;
  head_->trigger = opr_block;
  head_->write = true;

  // check if it is ready to write
  if (pending_write_ == nullptr) {
    // invariant: is_ready_to_read()
    pending_write_ = head_;
    CHECK_GE(num_pending_reads_, 0);
    if (num_pending_reads_ == 0) {
      // STATE CHANGE
      opr_block->decr_wait();
      num_pending_reads_ = kWriteTriggered;
    }
  } else {
    CHECK_NE(num_pending_reads_, 0);
  }
  head_ = new_var_block;
}
```

<a id="orgc165026"></a>

### CompleteReadDependency

如果一个  Read 依赖完成，只需要修改 `-- num_pending_reads` 来确保 `num_pending_reads` 表示了最新的 pending 的 Read 依赖的操作的数目。

如果所有 pending 的 Read 操作均已满足，则接着开始满足下一个 Write 的依赖，
如果 Write 依赖对应的 function 所有的参数依赖都已经完毕( `trigger->decr_wait() == 0` ) ， 则将其 dispatch 到执行引擎中实际执行。

```c++
template <typename Dispatcher>
inline void ThreadedVar::CompleteReadDependency(Dispatcher dispatcher) {
  OprBlock *trigger = nullptr;
  {
    // this is lock scope
    std::lock_guard<std::mutex> lock{m_};
    CHECK_GT(num_pending_reads_, 0);

    if (--num_pending_reads_ == 0) {
      if (pending_write_ != nullptr) {
        // STATE CHANGE
        trigger = pending_write_->trigger;
        num_pending_reads_ = kWriteTriggered;
      }
    }
  }
  if (trigger != nullptr && trigger->decr_wait() == 0) {
    dispatcher(trigger);
  }
}
```

<a id="org2c14aeb"></a>

### CompleteWriteDependency

由于 Write 依赖后面可能接了多个 Read 依赖，因此实现会复杂一些：

-   遍历链表知道找到下个 Write 依赖，用 `end_of_read_chain` 表示
-   每发现一个 Read 依赖就将 `num_pending_reads_ ++`
-   旧的 Write 依赖用指针 `old_pending_write` 表示， 两者之间全是 Read 依赖，while 循环并行满足其依赖

```c++
template <typename Dispatcher>
inline bool ThreadedVar::CompleteWriteDependency(Dispatcher dispatcher) {
  // this is lock scope
  VersionedVarBlock *old_pending_write, *end_of_read_chain;
  OprBlock* trigger_write = nullptr;
  {
    std::lock_guard<std::mutex> lock{m_};
    // invariants
    assert(head_->next == nullptr);
    assert(pending_write_ != nullptr);
    CHECK_EQ(num_pending_reads_, kWriteTriggered);

    // 删掉当前 Write 依赖的 VersionedVarBlock，快速返回
    if (to_delete_) {
      VersionedVarBlock *head = pending_write_->next;
      VersionedVarBlock::Delete(pending_write_);
      assert(head_ == head);
      VersionedVarBlock::Delete(head);
      return true;
    }
    // detach pending write
    old_pending_write = pending_write_;
    // search for chains to trigger
    end_of_read_chain = old_pending_write->next;
    // reset to 0 pending reads
    num_pending_reads_ = 0;
    while (end_of_read_chain != head_ &&
           end_of_read_chain->write == false) {
      ++num_pending_reads_;
      end_of_read_chain = end_of_read_chain->next;
    }
    if (end_of_read_chain == head_) {
      pending_write_ = nullptr;
    } else {
      // check if there is pending reads, if not trigger write
      assert(end_of_read_chain->write == true);
      pending_write_ = end_of_read_chain;
      if (num_pending_reads_ == 0) {
        // mark write as already activated in this var
        num_pending_reads_ = kWriteTriggered;
        trigger_write = end_of_read_chain->trigger;
      }
    }
  }
  // This is outside of lock scope
  // Be very carful, pending_write_ and num_pending_reads_
  // can change now, do not reply ont the two variables.
  // The linked list \in [old_pending_write, end_of_read_chain)
  // is already detached from this Var.
  // So it is safe to modify these
  VersionedVarBlock *cur_head = old_pending_write->next;
  VersionedVarBlock::Delete(old_pending_write);
  // dispatch all the events
  while (cur_head != end_of_read_chain) {
    if (cur_head->trigger->decr_wait() == 0) {
      dispatcher(cur_head->trigger);
    }
    auto prev = cur_head;
    cur_head = cur_head->next;
    assert(cur_head != nullptr);
    VersionedVarBlock::Delete(prev);
  }
  if (trigger_write != nullptr && trigger_write->decr_wait() == 0) {
    dispatcher(trigger_write);
  }
  return false;
}
```

<a id="orgb62c3f4"></a>

## Engine 总接口

首先给出存储 function 执行信息的 OprBlock，注意其中的 `wait` 字段表示，Opr 依赖的 Var 数目，当 `wait==0` 时，
表示所有的 Var 都可以满足了，此时对应的 function 就可以被 engine 真正执行了。

```c++
/*!
 * \brief Operation block in the scheduler.
 *  Each OprBlock corresponds to an operation pushed to the engine.
 */
struct OprBlock : public common::ObjectPoolAllocatable<OprBlock> {
  /*!
   * \brief wait number of pending tasks this OprBlock is waiting for.
   */
  std::atomic<int> wait{0};
  /*! \brief Pointer to information on performing real operation */
  ThreadedOpr* opr{nullptr};
  /*! \brief The context this operator */
  Context ctx;
  /*! \brief priority of the function */
  int priority;
  /*! \brief indicate whether to profile this operator */
  bool profiling{false};
  /*! \brief operator execution statistics */
  OprExecStat *opr_stat;
  // define possible debug information
  DEFINE_ENGINE_DEBUG_INFO(OprBlock);
  /*!
   * \brief call this function to decrease the wait counter.
   * \return the wait counter after the decreasement.
   */
  inline int decr_wait() {
    // chack invariant, avoid over trigger
    int ret = --wait;
    CHECK_GE(ret, 0);
    return ret;
  }
};  // struct OprBlock
```

总的调用接口，在 Push 一个 function 到 Engine 时

-   分析其 Var 的依赖关系，对 `const_vars` 和 `mutate_vars` 分别调用 `AppendReadDependency` 和 `AppendWriteDependency` 构建依赖关系
-   `opr_block->opr.wait` 记录依赖的参数数目
-   如果依赖直接满足，则执行之
    -   否则将任务丢到 engine 的队列中，进入异步的等待

```c++
void ThreadedEngine::Push(OprHandle op, Context exec_ctx, int priority, bool profiling) {
  ThreadedOpr* threaded_opr = ThreadedOpr::CastFromBase(op);
  OprBlock* opr_block = OprBlock::New();
  opr_block->opr = threaded_opr;

  opr_block->wait.store(static_cast<int>(
                                         threaded_opr->const_vars.size() +
                                         threaded_opr->mutable_vars.size() + 1));
  opr_block->ctx = exec_ctx;
  opr_block->priority = priority;
  opr_block->profiling = profiling;
  ++pending_;
  // Add read dependencies.
  for (auto&& i : threaded_opr->const_vars) {
    i->AppendReadDependency(opr_block);
  }
  // Add write dependencies.
  for (auto&& i : threaded_opr->mutable_vars) {
    i->AppendWriteDependency(opr_block);
  }
  if (opr_block->decr_wait() == 0) {
    this->PushToExecute(opr_block, true);
  }
}
```

其中负责 function 的具体执行的是 `PushToExecute` 函数，其具体实现有两种：

-   `threaded_engine_pooled.cc` 所有 device 共用一个 pool 的实现
-   `threaded_engine_perdevice.cc` 区分 device 的 engine


<a id="org40885ae"></a>

## ThreadedEnginePooled

这里的实现比 `ThreadedEnginePerDevice` 简单一些，大概逻辑是：

-   维护 2 个并发的任务队列，一个为 IO 任务， 一个为非 IO 任务
-   如果是 `pusher_thread` 的 function，则立即执行，否则添加到对应的任务队列中

```c++
void PushToExecute(OprBlock *opr_block, bool pusher_thread) override {
  if (opr_block->opr->prop == FnProperty::kAsync && pusher_thread) {
    DoExecute(opr_block);
  } else {
    DoPushToQueue(opr_block);
  }
}
```

这里 `pusher_thread` ，如果为 true 则立即执行，否则添加到任务队列里，注意到 上小节中 engine 中 `Push` 中如此调用：

```c++
if (opr_block->decr_wait() == 0) {
  this->PushToExecute(opr_block, true);
}
```

就是对 Var 依赖的 `opr_block` 会首先被处理（check 依赖是否被满足啥的）。

mxnet 通过 engine.h 中定义的 `FnProperty` 将 function 分为以下 5 种

```c++
enum class FnProperty {
  /*! \brief Normal operation */
  kNormal,
  /*! \brief Copy operation from GPU to other devices */
  kCopyFromGPU,
  /*! \brief Copy operation from CPU to other devices */
  kCopyToGPU,
  /*! \brief Prioritized sync operation on CPU */
  kCPUPrioritized,
  /*! \brief Asynchronous function call */
  kAsync
};  // enum class FnProperty
```

不同的任务类型对计算/IO 资源的占用情况不同，会有不同的队列负责执行。

在 `ThreadedEnginePooled` 中安是否是 IO 任务将并发任务队列拆成：

1.  `io_task_queue` , 负责 kCopyFromGPU, kCopyToGPU
2.  `task_queue` , 所有其他的类型

于是有 `DoPushToQueue` 中的实现：

```c++
/*!
 * \brief Push the operation to the queue.
 * \param opr_block The operator block.
 */
void DoPushToQueue(OprBlock* opr_block) {
  switch (opr_block->opr->prop) {
  case FnProperty::kCopyFromGPU:
  case FnProperty::kCopyToGPU: {
    io_task_queue_.Push(opr_block);
    break;
  }
  default: {
    task_queue_.Push(opr_block);
    break;
  }
}
```

而两个任务队列的实现和线程池的细节如下：

```c++
dmlc::ConcurrentBlockingQueue<OprBlock*> task_queue_;
dmlc::ConcurrentBlockingQueue<OprBlock*> io_task_queue_;

ThreadPool thread_pool_;
ThreadPool io_thread_pool_;

void ThreadWorker(dmlc::ConcurrentBlockingQueue<OprBlock*>* task_queue) {
  OprBlock* opr_block;
  while (task_queue->Pop(&opr_block)) {
    DoExecute(opr_block);
  }
}
```

这里的线程池就是  `engine/thread_pool.h` 中的实现。


<a id="orgc0e354d"></a>

## ThreadedEnginePerDevice

`ThreadedEnginePerDevice` 在 `ThreadedEngine` 的基础之上支持如下功能：

-   每个 device（GPU 卡/CPU 核？) 固定数目的线程数
-   对 IO 操作和高优先级操作分配不同的任务队列
-   针对 GPU，每个线程使用单独的 stream，互不影响

四个任务队列：

```c++
common::LazyAllocArray<ThreadWorkerBlock<kWorkerQueue> > cpu_normal_workers_;
// cpu priority worker
std::unique_ptr<ThreadWorkerBlock<kPriorityQueue> > cpu_priority_worker_;
// workers doing normal works on GPU
common::LazyAllocArray<ThreadWorkerBlock<kWorkerQueue> > gpu_normal_workers_;
// workers doing copy works from/to GPU
common::LazyAllocArray<ThreadWorkerBlock<kCopyQueue> > gpu_copy_workers_;
```

这里 `gpu_copy_workers_` 对应着 IO 操作的任务队列，
`cpu_normal_workers_` , `gpu_normal_workers_`  和 `gpu_copy_workers_` 均为每个 device 单独分配线程池。
`cpu_priority_worker_` 不区分 device.

`cpu_priority_worker_` 不区分 device 的目的是，利用所有的 CPU device 资源优先执行这些高优先的任务（类似常规的 CPU 多核并行程序），
而其他线程池区分 device 的目的是，各个 device 资源的追踪和充分利用，特别对于 GPU 这类。

其中类型 `ThreadWorkerBlock` 打包了 Queue 和 ThreadPool:

```c++
template<dmlc::ConcurrentQueueType type>
struct ThreadWorkerBlock {
  // task queue on this task
  dmlc::ConcurrentBlockingQueue<OprBlock*, type>  task_queue;
  // thread pool that works on this task
  std::unique_ptr<ThreadPool> pool;
  // destructor
  ~ThreadWorkerBlock() noexcept(false) {
    task_queue.SignalForKill();
  }
};
```

主体接口 `PushToExecute` 和 `ThreadedEngine` 中的实现的逻辑类似：

```c++
void PushToExecute(OprBlock *opr_block, bool pusher_thread) override {
  const Context& ctx = opr_block->ctx;
  // pusher_thread 直接执行
  if (opr_block->opr->prop == FnProperty::kAsync && pusher_thread) {
    if (ctx.dev_mask() == gpu::kDevMask) {
      #if MXNET_USE_CUDA
      MSHADOW_CATCH_ERROR(mshadow::SetDevice<gpu>(ctx.dev_id));
      #endif
    }
    RunContext run_ctx;
    run_ctx.stream = nullptr;
    this->ExecuteOprBlock(run_ctx, opr_block);
  } else {
    // cpu 模式
    if (ctx.dev_mask() == cpu::kDevMask) {
      // 如果是高优先级任务， 在 cpu_priority_worker_ 中执行
      // 该队列不区分 device，在 CPU 多核上并发执行（空间 device 优先执行之）
      if (opr_block->opr->prop == FnProperty::kCPUPrioritized) {
        cpu_priority_worker_->task_queue.Push(opr_block, opr_block->priority);
      } else {
        // 否则乖乖仔 cpu_normal_workers_ 中分 device 执行
        // 每个核会有自己的 thread pool ?
        int dev_id = ctx.dev_id;
        int nthread = cpu_worker_nthreads_;
        cpu_normal_workers_.Get(dev_id, [this, dev_id, nthread]() {
            auto blk = new ThreadWorkerBlock<kWorkerQueue>();
            blk->pool.reset(new ThreadPool(nthread, [this, blk] () {
                  this->CPUWorker(blk);
                }));
            return blk;
          })->task_queue.Push(opr_block, opr_block->priority);
      }
      // GPU 模式
    } else {
      CHECK_EQ(ctx.dev_mask(), gpu::kDevMask);
      // GPU execution.
      FnProperty prop = opr_block->opr->prop;
      bool is_copy = (prop == FnProperty::kCopyFromGPU ||
                      prop == FnProperty::kCopyToGPU);
      int nthread = gpu_worker_nthreads_;
      int dev_id = ctx.dev_id;
      // IO 的 copy 操作，CPU <-> GPU 代价较大，需要单独线程异步去做
      // 默认 1 个 device 上只分配 1 个 IO 线程，因为此处多线程拷贝也没效果
      if (is_copy) {
        gpu_copy_workers_.Get(dev_id, [this, dev_id, is_copy, nthread]() {
            auto blk = new ThreadWorkerBlock<kCopyQueue>();
            blk->pool.reset(new ThreadPool(nthread, [this, dev_id, is_copy, blk] () {
                  this->GPUWorker(dev_id, is_copy, blk);
                }));
            return blk;
          })->task_queue.Push(opr_block, opr_block->priority);
      } else {
        // 是计算任务，则提交到 gpu 的计算队列中
        gpu_normal_workers_.Get(dev_id, [this, dev_id, is_copy, nthread]() {
            auto blk = new ThreadWorkerBlock<kWorkerQueue>();
            blk->pool.reset(new ThreadPool(nthread, [this, dev_id, is_copy, blk] () {
                  this->GPUWorker(dev_id, is_copy, blk);
                }));
            return blk;
          })->task_queue.Push(opr_block, opr_block->priority);
      }
    }
  }
}
```

其他实现基本跟 `ThreadedEnginePooled` 里的一致，最后给出 `GPUWorker` 的实现：

```c++
template<dmlc::ConcurrentQueueType type>
inline void GPUWorker(int dev_id,
                      bool is_copy_worker,
                      ThreadWorkerBlock<type> *block) {
#if MXNET_USE_CUDA
  // allocate stream
  mshadow::SetDevice<gpu>(dev_id);
  RunContext run_ctx;
  mshadow::Stream<gpu> *stream;
  // 每个 GPUWorker 会分配自己的 stream
  // 如果是 IO 的操作，直接分配显存
  // 如果是正常的计算，则会按计算的方法分别分配 blas 活 cudnn 对应的显存
  if (is_copy_worker) {
    stream = mshadow::NewStream<gpu>(false, false);
  } else {
    stream = mshadow::NewStream<gpu>(true, MXNET_USE_CUDNN != 0);
  }
  run_ctx.stream = stream;
  // execute task
  OprBlock* opr_block;
  auto* task_queue = &(block->task_queue);
  while (task_queue->Pop(&opr_block)) {
    this->ExecuteOprBlock(run_ctx, opr_block);
  }
  // Catch exception for CUDA driver shutdown
  MSHADOW_CATCH_ERROR(mshadow::DeleteStream<gpu>(stream));
#endif
}
```

<a id="orgcdbea0b"></a>

# 参考文献

1.  [Dependency Engine for Deep Learning](http://mxnet.io/architecture/note_engine.html)
2.  [mxnet dep engine implemention](https://github.com/dmlc/mxnet/blob/b11d3a2550b3ad9d96f42e7d15e2c418dd2b4c52/docs/zh/mxnet-dep-engine-implemention.md)


