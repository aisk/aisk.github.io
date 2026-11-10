% 用 GGBond 学 GGML（一）：Backend、Scheduler 和一次矩阵乘法

本文是“用 GGBond 学 GGML”系列的第一篇。这个系列借助 [GGBond](https://github.com/aisk/ggbond) 这个 Python 封装来介绍 GGML 的概念和用法，后续文章会依次涉及张量与形状、计算图与内存分配、GGUF 模型文件的加载，最后实现一个完整模型的推理。本篇介绍写这个系列的原因、GGML 的两阶段计算方式，以及几套 API 之间的关系，并以矩阵乘法为例给出可以运行的代码。

## GGML 的学习门槛

[GGML](https://github.com/ggml-org/ggml) 是 llama.cpp 和 whisper.cpp 使用的张量库。它体积小，几乎没有外部依赖，支持 CPU、CUDA、Metal、Vulkan 等多种设备，适合用来自行实现模型推理。但是它的学习门槛比较高，原因主要有三个。

首先，它缺少教程。官方文档基本上只有头文件里的注释和 `examples` 目录下的几个示例程序。要了解某个函数的用法，通常只能去 llama.cpp 的源码里查找它的调用。

其次是语言。GGML 的对外接口是纯 C，内部实现是 C 和 C 风格的 C++。使用者需要手动管理内存，需要自己计算缓冲区的大小，参数有误时 `GGML_ASSERT` 会直接终止进程。对于主要使用 Python 的开发者，这些都增加了上手的难度。

最后，GGML 在演进过程中积累了多套 API。它们目前都可以使用，也都出现在头文件和示例里，但其中一部分在多数场景下已经没有必要使用。初学者面对同一件事的多种写法，不容易判断应该学习哪一种，也不容易看出它们之间的关系。

## GGBond

GGBond 是 GGML 的一个 Python 封装，它在概念上与 GGML 基本一一对应。

GGML 的设计本身接近面向对象，只是受 C 语言的限制，表现为一组以某个结构体为第一个参数的函数，例如 `ggml_new_tensor_2d(ctx, ...)` 和 `ggml_backend_graph_compute(backend, graph)`。GGBond 把这些结构体封装成类，把函数改为方法，方法名沿用原函数名去掉前缀后的部分。上面两个调用在 GGBond 中分别是 `ctx.new_tensor_2d(...)` 和 `backend.compute(graph)`。

在此基础上，GGBond 增加了少量 Python 语言层面的便利。资源可以用 `with` 语句自动释放，张量数据通过 numpy 数组读写，一部分会导致进程终止的断言被提前检查并转换为 Python 异常。除此之外它不引入 GGML 没有的概念，建图、分配内存、执行计算这些步骤仍然需要显式写出，每一步都有对应的 C 函数。

这些特点使它适合用于学习 GGML。

- 不需要学习新的语言，也不需要配置 C/C++ 的工程环境。
- 可以在 REPL 中逐步执行并观察结果，例如创建张量后查看它的形状，建图后查看图中的节点。
- 概念和函数名与 GGML 一致，所以学到的内容可以直接用于阅读 llama.cpp 的源码，或者用 C/C++ 编写 GGML 程序。

GGBond 也可以用于实际的开发。一种做法是先用它验证模型实现的正确性，再改写为等价的 C/C++ 代码。另一种做法是直接使用 Python 代码。实际的计算在 GGML 内部完成，运行在 CPU 线程或显卡上，Python 一侧只负责加载权重、建图和发起计算，这部分开销在整个推理过程中占比很小，因此性能一般不是主要问题。

它的不足在于分发。编译好的 GGML 程序是单个可执行文件，而使用 GGBond 的程序需要附带 Python 运行时和 GGML 的动态库。不过与 PyTorch 相比，需要附带的内容仍然少得多。

GGBond 目前需要从源码安装。它本身是纯 Python 代码，但依赖的 [ggml-python](https://github.com/abetlen/ggml-python) 会在安装时编译 GGML，因此需要系统中有 C/C++ 编译器和 CMake。

```bash
pip install git+https://github.com/aisk/ggbond
```

## 两阶段计算

GGML 的计算分为两个阶段，先定义计算，再执行计算。

在 PyTorch 中，`c = a @ b` 执行之后 `c` 中就有了结果。GGML 的行为不同。对张量调用运算函数只是描述了这次运算，返回的新张量记录了运算的类型和输入，其中没有数据。下面在 REPL 中演示这一点。

```python
>>> from ggbond.ggml import Context, F32
>>> ctx = Context(1 << 20, no_alloc=True)
>>> a = ctx.new_tensor_2d(F32, 2, 4, name="a")
>>> b = ctx.new_tensor_2d(F32, 2, 3, name="b")
>>> c = ctx.mul_mat(a, b)
>>> c
<ggml.Tensor name='' ne=(4, 3, 1, 1) dtype=F32>
>>> c.op
<Op.MUL_MAT: 29>
>>> c.src
(<ggml.Tensor name='a' ne=(2, 4, 1, 1) dtype=F32>, <ggml.Tensor name='b' ne=(2, 3, 1, 1) dtype=F32>)
```

此时 `c` 的形状、运算类型和输入都已确定，但计算还没有发生。

下一步是从结果张量出发，把它依赖的所有张量收集为一张计算图。

```python
>>> graph = ctx.new_graph().build_forward_expand(c)
>>> graph
<ggml.Graph n_nodes=1 n_leafs=2>
>>> list(graph.nodes())
[<ggml.Tensor name='node_0' ne=(4, 3, 1, 1) dtype=F32>]
>>> list(graph.leafs())
[<ggml.Tensor name='a' ne=(2, 4, 1, 1) dtype=F32>, <ggml.Tensor name='b' ne=(2, 3, 1, 1) dtype=F32>]
```

图中有一个需要计算的节点和两个叶子。叶子是不需要计算、只提供数据的张量。至此第一阶段完成。第二阶段是为图中的张量分配内存并执行这张图，执行之后 `c` 中才有数据。

采用这种方式，是因为 GGML 在执行前就能得到完整的计算过程。它可以据此提前算出所需的内存总量，让生命周期不重叠的中间结果共用同一块内存，并决定图的各个部分分别在哪个设备上执行。

GGML 的几套 API 在第一阶段没有区别，建图的方式相同。它们的区别在第二阶段，也就是张量数据存放在哪里，以及由谁执行计算图。

## Context、Backend 和 Scheduler

GGML 最初只支持 CPU，目标是利用 AVX2、NEON 等向量指令集加速矩阵运算。在当时的设计中，一个 `ggml_context` 对应一块内存，张量的元信息、张量的数据和计算图都存放在其中。创建张量时会同时分配数据的空间，计算也在这块内存上进行。这种用法通常称为 Context API。它目前仍然可用，在早期的示例和文章中也很常见，但已经不推荐使用，本系列不会采用它。这里提到它，是因为了解这段历史有助于理解 context 现在的职责。

**Backend API** 是为支持显卡而引入的。显卡上的数据位于显存，无法与元信息一起存放在 context 的主机内存里。GGML 因此把张量数据的分配和存放从 context 中分离出来，交给新增的 backend。Context 保留了下来，但只负责存放元信息和计算图，创建时需要指定 `no_alloc=true`，表示不为张量数据分配空间。数据存放在 backend 分配的 buffer 中，计算也由 backend 执行。在这套体系中 CPU 同样是一个 backend，与 CUDA、Metal 的地位相同，它使用的仍然是原有的 CPU 计算代码，因此性能与 Context API 没有差别。

**Scheduler API** 建立在 Backend API 之上，用于让一张图在多个 backend 上执行。常见的场景有两种。一种是显卡 backend 不支持图中的某个运算，这部分需要由 CPU 执行。另一种是显存不足以容纳整个模型，只能把一部分层放在显卡上。Scheduler 根据每个节点可以在哪些 backend 上执行，以及权重所在的 backend，把图划分为若干段，每一段交给一个 backend，并在相邻的段之间拷贝数据。需要说明的是，这些段是按顺序依次执行的，多个设备并不会同时计算。

| | 张量数据的位置 | 负责分配 | 负责计算 |
| --- | --- | --- | --- |
| Backend API | 一个 backend 的 buffer | `ggml_backend_alloc_ctx_tensors` 和 `ggml_gallocr` | `ggml_backend_graph_compute` |
| Scheduler API | 多个 backend 的 buffer | `ggml_backend_sched` | `ggml_backend_sched_graph_compute` |

llama.cpp 和 whisper.cpp 目前都基于 Scheduler API，在只使用一个 backend 的情况下也是如此，它是现在的主要用法。Backend API 是理解 Scheduler 的基础，在确定只使用一个设备时也可以直接使用。

下面用同一个例子分别演示这两套 API。

## 示例：矩阵乘法

示例使用 GGML 自带的 `simple` 示例中的数据，一个 4×2 的矩阵和一个 3×2 的矩阵。

```python
import numpy as np

a = np.array([[2, 8], [5, 1], [4, 2], [8, 6]], dtype=np.float32)  # 4 x 2
b = np.array([[10, 5], [9, 9], [5, 4]], dtype=np.float32)         # 3 x 2
```

这里需要先说明形状的问题。GGML 的维度顺序与 numpy 相反，numpy 中 4 行 2 列的矩阵在 GGML 中要写成 `new_tensor_2d(F32, 2, 4)`，列数在前，行数在后。`mul_mat` 的语义也因此与 numpy 的 `@` 不同，`mul_mat(a, b)` 相当于 numpy 中的 `b @ a.T`。所以这个示例的结果是一个 3×4 的矩阵。

```python
>>> b @ a.T
array([[ 60.,  55.,  50., 110.],
       [ 90.,  54.,  54., 126.],
       [ 42.,  29.,  28.,  64.]], dtype=float32)
```

形状的规则会在下一篇中详细介绍，本篇只需要知道维度顺序是相反的。

### Backend API

```python
from ggbond.ggml import Backend, Context, GAllocr, F32

with Backend.cpu_init() as backend, \
        Context(1 << 20, no_alloc=True) as ctx_w, \
        Context(1 << 20, no_alloc=True) as ctx_g:
    # the "weights": only metadata is created in ctx_w
    ta = ctx_w.new_tensor_2d(F32, 2, 4, name="a")
    tb = ctx_w.new_tensor_2d(F32, 2, 3, name="b")

    # ggml_backend_alloc_ctx_tensors: allocate their data on the backend
    backend.alloc_ctx_tensors(ctx_w)
    ta.write(a)
    tb.write(b)

    # phase 1: define, in a separate context
    tc = ctx_g.mul_mat(ta, tb)
    graph = ctx_g.new_graph().build_forward_expand(tc)

    # phase 2: allocate the graph's tensors, then compute
    with GAllocr.from_backend(backend) as allocr:
        allocr.alloc_graph(graph)
        backend.compute(graph)

        print(tc.to_numpy())
```

```
[[ 60.  55.  50. 110.]
 [ 90.  54.  54. 126.]
 [ 42.  29.  28.  64.]]
```

代码中用到了 `Backend`、`Context` 和 `GAllocr` 三种对象。

`Backend` 表示存放数据和执行计算的设备。这里使用的是 CPU。改用 `Backend.cuda_init()` 或 `Backend.metal_init()` 时，其余代码不需要修改。

两个 `Context` 都指定了 `no_alloc=True`，各自的 1MB 空间只用于存放元信息，与数据的大小无关。`ctx_w` 存放长期存在的张量，在实际的模型中对应权重。`ctx_g` 存放计算图和图中的中间张量。GGML 程序通常会把二者分开，因为权重只需要加载一次，而计算图可能在每次推理时重新构建。`ctx_g.mul_mat(ta, tb)` 表示运算结果创建在 `ctx_g` 中，而两个输入来自 `ctx_w`，这与 C 中 `ggml_mul_mat(ctx, a, b)` 第一个参数的含义相同。

张量数据的空间需要显式分配，分配方式有两种。

- `backend.alloc_ctx_tensors(ctx_w)` 为一个 context 中已有的全部张量在 backend 上分配空间，适用于权重这类长期存在的张量。分配之后才能调用 `write`，写入的数据会传输到 backend，使用显卡时即传输到显存。
- `GAllocr` 对应 `ggml_gallocr`，为一张图中需要计算的张量分配空间。它会分析整张图，让后面的节点复用已经不再需要的中间结果所占的内存，因此总的内存占用远小于为每个张量单独分配的情况。

`GAllocr` 分配的内存由它自己持有，关闭后即被释放，所以结果需要在它关闭之前读取。示例中 `tc.to_numpy()` 位于 `with` 块内部就是出于这个原因。

### Scheduler API

```python
from ggbond.ggml import Backend, Context, DeviceType, Scheduler, F32

Backend.load_all()

with Backend.init_best() as gpu, \
        Backend.init_by_type(DeviceType.CPU) as cpu, \
        Context(1 << 20, no_alloc=True) as ctx, \
        Scheduler(gpu, cpu) as sched:
    # phase 1: define, inputs are part of the graph this time
    ta = ctx.new_tensor_2d(F32, 2, 4, name="a").set_input()
    tb = ctx.new_tensor_2d(F32, 2, 3, name="b").set_input()
    tc = ctx.mul_mat(ta, tb).set_output()
    graph = ctx.new_graph().build_forward_expand(tc)

    # phase 2: the scheduler allocates and computes
    sched.alloc_graph(graph)
    ta.write(a)
    tb.write(b)
    sched.compute(graph)

    print(tc.to_numpy())
    print(sched.n_splits, sched.get_tensor_backend(tc))
```

```
[[ 60.  55.  50. 110.]
 [ 90.  54.  54. 126.]
 [ 42.  29.  28.  64.]]
1 <ggml.Backend name='CPU'>
```

`Backend.load_all()` 注册当前 GGML 构建所支持的全部 backend，`Backend.init_best()` 从中选择最合适的设备，存在显卡时会选择显卡。`Scheduler(gpu, cpu)` 按优先级顺序接收多个 backend，靠前的优先使用。最后一个通常是 CPU，用于执行其他 backend 不支持的运算。

Scheduler 内部包含了图的分配器，因此不再需要 `GAllocr`。`sched.alloc_graph` 和 `sched.compute` 分别取代了上一个示例中的 `allocr.alloc_graph` 和 `backend.compute`。

这个示例提供输入的方式也与上一个不同。`ta` 和 `tb` 没有放在单独的 context 中预先分配，而是作为图的输入，用 `set_input()` 标记，与中间结果一样由 Scheduler 分配，所以写入数据的操作位于 `alloc_graph` 之后。`set_output()` 表示该张量是最终结果，分配器不会让其他节点复用它的内存。在实际的模型中这两种方式会同时使用，权重通过 `alloc_ctx_tensors` 预先分配并上传，每次推理都会变化的输入（例如 token）使用 `set_input()`。

最后一行输出的是图被划分的段数，以及 `tc` 被分配到的 backend。本文运行环境中的 GGML 只包含 CPU backend，`init_best()` 返回的也是 CPU，所以图只有一段。在有显卡的环境中，同样的代码会在显卡上执行这次乘法。

## 小结

- GGML 的计算分为两个阶段。对张量调用运算函数只是定义计算图，分配内存并执行之后才会得到结果。
- Context 负责存放元信息和计算图。早期它还负责存放数据和执行计算，这种用法已经不推荐。
- Backend 负责存放数据和执行计算。数据的空间需要显式分配，长期存在的张量使用 `alloc_ctx_tensors`，图中的张量使用 `GAllocr`。
- Scheduler 把一张图分配给多个 backend 执行，并包含了图的分配器，是目前的主要用法。

下一篇介绍张量，包括维度顺序和形状的规则。
