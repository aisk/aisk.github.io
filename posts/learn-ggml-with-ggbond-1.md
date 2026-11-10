% Learning GGML with GGBond, Part 1: Backend, Scheduler, and a Matrix Multiplication

This is the first post in a series called "Learning GGML with GGBond". The series uses [GGBond](https://github.com/aisk/ggbond), a Python wrapper, to introduce the concepts and usage of GGML. Later posts will cover tensors and shapes, computation graphs and memory allocation, loading GGUF model files, and finally the inference of a complete model. This post explains why I am writing the series, how GGML's two-phase computation works, and how its several APIs relate to each other. It ends with a runnable matrix multiplication example.

## Why GGML is hard to learn

[GGML](https://github.com/ggml-org/ggml) is the tensor library used by llama.cpp and whisper.cpp. It is small, has almost no external dependencies, and supports CPU, CUDA, Metal, Vulkan and other devices, which makes it a good choice for implementing model inference yourself. However, it is fairly hard to learn, for three main reasons.

First, there are very few tutorials. The official documentation is mostly the comments in the header files plus a handful of programs in the `examples` directory. To find out how a function is used, you usually have to search for its call sites in the llama.cpp source.

Second, there is the language. The public interface of GGML is plain C, and the implementation is C and C-style C++. You have to manage memory by hand and compute buffer sizes yourself, and a wrong argument makes `GGML_ASSERT` terminate the process. For developers who mainly work in Python, all of this raises the cost of getting started.

Third, GGML has accumulated several APIs over time. All of them still work and all of them appear in the headers and examples, but some are no longer necessary in most situations. A beginner who sees several ways to do the same thing cannot easily tell which one to learn, or how they relate to each other.

## GGBond

GGBond is a Python wrapper for GGML. Its concepts map almost one to one onto those of GGML.

The design of GGML is already close to object-oriented. Because it is written in C, it appears as a set of functions that take some struct as their first argument, such as `ggml_new_tensor_2d(ctx, ...)` and `ggml_backend_graph_compute(backend, graph)`. GGBond wraps these structs as classes and turns the functions into methods, and each method keeps the name of the original function without its prefix. The two calls above become `ctx.new_tensor_2d(...)` and `backend.compute(graph)` in GGBond.

On top of that, GGBond adds a small amount of convenience from the Python language. Resources can be released automatically with the `with` statement, tensor data is read and written as numpy arrays, and some of the assertions that would terminate the process are checked in advance and raised as Python exceptions. Beyond this it introduces no concept that GGML does not have. Building a graph, allocating memory and running the computation still have to be written out explicitly, and each step has a corresponding C function.

These properties make it suitable for learning GGML.

- There is no new language to learn and no C/C++ build environment to set up.
- You can run code step by step in the REPL and inspect the results, for example the shape of a tensor after creating it, or the nodes of a graph after building it.
- The concepts and function names are the same as in GGML, so what you learn applies directly to reading the llama.cpp source or writing GGML programs in C/C++.

GGBond can also be used for real development. One approach is to use it to verify that a model implementation is correct, and then rewrite it as equivalent C/C++ code. Another is to use the Python code as it is. The actual computation happens inside GGML, on CPU threads or on the GPU. The Python side only loads weights, builds graphs and starts the computation, which is a small part of the total inference cost, so performance is usually not the main concern.

Its weakness is distribution. A compiled GGML program is a single executable, while a program using GGBond has to ship with a Python runtime and the GGML shared libraries. Compared with PyTorch, though, this is still far less to ship.

For now GGBond has to be installed from source. GGBond itself is pure Python, but its dependency [ggml-python](https://github.com/abetlen/ggml-python) compiles GGML during installation, so a C/C++ compiler and CMake are required.

```bash
pip install git+https://github.com/aisk/ggbond
```

## Two-phase computation

Computation in GGML has two phases. The computation is defined first and executed afterwards.

In PyTorch, `c` holds the result as soon as `c = a @ b` has run. GGML behaves differently. Calling an operation on tensors only describes that operation. The returned tensor records the type of the operation and its inputs, and it contains no data. The following REPL session shows this.

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

At this point the shape, the operation and the inputs of `c` are known, but nothing has been computed.

The next step starts from the result tensor and collects every tensor it depends on into a computation graph.

```python
>>> graph = ctx.new_graph().build_forward_expand(c)
>>> graph
<ggml.Graph n_nodes=1 n_leafs=2>
>>> list(graph.nodes())
[<ggml.Tensor name='node_0' ne=(4, 3, 1, 1) dtype=F32>]
>>> list(graph.leafs())
[<ggml.Tensor name='a' ne=(2, 4, 1, 1) dtype=F32>, <ggml.Tensor name='b' ne=(2, 3, 1, 1) dtype=F32>]
```

The graph has one node to compute and two leafs. A leaf is a tensor that is not computed and only provides data. This completes the first phase. The second phase allocates memory for the tensors in the graph and executes it, and only then does `c` contain data.

GGML works this way because it can see the whole computation before running it. With that information it can work out the total memory required in advance, let intermediate results whose lifetimes do not overlap share the same memory, and decide which device executes each part of the graph.

The APIs of GGML do not differ in the first phase, since the graph is built in the same way. They differ in the second phase, that is, in where tensor data is stored and in what executes the graph.

## Context, Backend and Scheduler

GGML originally supported only the CPU, and its goal was to speed up matrix operations with vector instruction sets such as AVX2 and NEON. In that design a `ggml_context` corresponds to one block of memory that holds the tensor metadata, the tensor data and the computation graph. Creating a tensor also allocates space for its data, and the computation runs on that same memory. This style is usually called the Context API. It still works and is common in older examples and articles, but it is no longer recommended, and this series does not use it. I mention it because knowing this history helps explain what a context is responsible for today.

The **Backend API** was introduced to support GPUs. Data on a GPU lives in GPU memory and cannot be stored alongside the metadata in the host memory of a context. GGML therefore separated the allocation and storage of tensor data from the context and handed them to a new object, the backend. The context remains, but it only holds metadata and computation graphs. It is created with `no_alloc=true`, which means that no space is allocated for tensor data. The data is stored in buffers allocated by a backend, and the backend also executes the computation. The CPU is a backend in this model as well, on equal footing with CUDA and Metal. It runs the same CPU code as before, so its performance is no different from that of the Context API.

The **Scheduler API** is built on the Backend API and lets one graph run on several backends. There are two common cases. In one, the GPU backend does not support some operation in the graph, and that part has to run on the CPU. In the other, GPU memory cannot hold the whole model, so only some of the layers are placed on the GPU. The scheduler looks at which backends can execute each node and at which backend holds the weights, divides the graph into several splits, assigns each split to one backend, and copies data between adjacent splits. Note that the splits are executed one after another. The devices do not compute at the same time.

| | Where tensor data lives | What allocates it | What computes |
| --- | --- | --- | --- |
| Backend API | Buffers of one backend | `ggml_backend_alloc_ctx_tensors` and `ggml_gallocr` | `ggml_backend_graph_compute` |
| Scheduler API | Buffers of several backends | `ggml_backend_sched` | `ggml_backend_sched_graph_compute` |

llama.cpp and whisper.cpp are both built on the Scheduler API today, even when only one backend is in use, and it is now the main way to use GGML. The Backend API is the basis for understanding the scheduler, and it can also be used directly when you know that only one device is involved.

The rest of this post shows the same example with each of these two APIs.

## Example: matrix multiplication

The example uses the data from GGML's own `simple` example, a 4×2 matrix and a 3×2 matrix.

```python
import numpy as np

a = np.array([[2, 8], [5, 1], [4, 2], [8, 6]], dtype=np.float32)  # 4 x 2
b = np.array([[10, 5], [9, 9], [5, 4]], dtype=np.float32)         # 3 x 2
```

Shapes need a short explanation first. GGML orders dimensions in the reverse of numpy. A matrix with 4 rows and 2 columns in numpy is written as `new_tensor_2d(F32, 2, 4)` in GGML, with the number of columns before the number of rows. The semantics of `mul_mat` therefore also differ from numpy's `@`, and `mul_mat(a, b)` is equivalent to `b @ a.T` in numpy. The result of this example is a 3×4 matrix.

```python
>>> b @ a.T
array([[ 60.,  55.,  50., 110.],
       [ 90.,  54.,  54., 126.],
       [ 42.,  29.,  28.,  64.]], dtype=float32)
```

The rules for shapes are covered in detail in the next post. For this one it is enough to know that the order of dimensions is reversed.

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

The code uses three kinds of objects, `Backend`, `Context` and `GAllocr`.

A `Backend` represents the device that stores the data and executes the computation. Here it is the CPU. Switching to `Backend.cuda_init()` or `Backend.metal_init()` requires no change to the rest of the code.

Both `Context` objects are created with `no_alloc=True`. The 1 MB given to each is used only for metadata and is unrelated to the size of the data. `ctx_w` holds long-lived tensors, which correspond to the weights in a real model. `ctx_g` holds the computation graph and the intermediate tensors in it. GGML programs usually keep the two apart, because the weights are loaded once while the graph may be rebuilt for every inference. `ctx_g.mul_mat(ta, tb)` means that the result is created in `ctx_g` while both inputs come from `ctx_w`, which is what the first argument of `ggml_mul_mat(ctx, a, b)` means in C.

The space for tensor data has to be allocated explicitly, and there are two ways to do it.

- `backend.alloc_ctx_tensors(ctx_w)` allocates space on the backend for all tensors that currently exist in a context. It suits long-lived tensors such as weights. `write` can only be called after this allocation, and the data written is transferred to the backend, which means GPU memory when a GPU is used.
- `GAllocr` corresponds to `ggml_gallocr` and allocates space for the tensors a graph has to compute. It analyses the whole graph and lets later nodes reuse the memory of intermediate results that are no longer needed, so the total memory use is much lower than allocating every tensor separately.

The memory allocated by a `GAllocr` is owned by it and is released when it is closed, so results must be read before that. This is why `tc.to_numpy()` is inside the `with` block in the example.

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

`Backend.load_all()` registers every backend that the current GGML build supports, and `Backend.init_best()` picks the most suitable device among them, which is a GPU when one is present. `Scheduler(gpu, cpu)` takes several backends in priority order, and earlier ones are preferred. The last one is usually the CPU, which executes the operations that the other backends do not support.

The scheduler contains a graph allocator of its own, so `GAllocr` is no longer needed. `sched.alloc_graph` and `sched.compute` replace `allocr.alloc_graph` and `backend.compute` from the previous example.

This example also provides its inputs differently. `ta` and `tb` are not placed in a separate context and allocated in advance. They are inputs of the graph, marked with `set_input()`, and the scheduler allocates them together with the intermediate results, which is why the data is written after `alloc_graph`. `set_output()` marks a tensor as a final result, so the allocator does not let other nodes reuse its memory. A real model uses both approaches together. The weights are allocated in advance with `alloc_ctx_tensors` and uploaded once, and inputs that change with every inference, such as tokens, use `set_input()`.

The last line prints the number of splits the graph was divided into and the backend that `tc` was assigned to. The GGML build used for this post only includes the CPU backend, so `init_best()` also returns the CPU and the graph has a single split. On a machine with a GPU, the same code runs the multiplication on the GPU.

## Summary

- Computation in GGML has two phases. Calling an operation on tensors only defines the computation graph, and the result exists only after memory has been allocated and the graph has been executed.
- A context holds metadata and computation graphs. In the past it also stored data and executed the computation, a usage that is no longer recommended.
- A backend stores data and executes the computation. The space for data is allocated explicitly, with `alloc_ctx_tensors` for long-lived tensors and `GAllocr` for the tensors in a graph.
- A scheduler distributes one graph over several backends and includes the graph allocator. It is the main way to use GGML today.

The next post covers tensors, including the order of dimensions and the rules for shapes.
