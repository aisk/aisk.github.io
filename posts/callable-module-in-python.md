% Making Python Modules Callable: Introducing Cadule

When writing Python code, I often find myself missing a feature from Node.js: the ability to directly `require` a module and have it become callable. In Node.js, you can export a function from a module and call it immediately:

```javascript
const messUpThings = require('./mess-up-things');
messUpThings(); // Works!
```

I know some developers aren't fond of this pattern, but there are legitimate use cases for it.

## The Problem

Consider a scenario where you need to write a helper function with a self-descriptive name, like `mess_up_things`. The function name itself tells you what it does. You have a few options:

1. **Create a util or helper module**: Put it in a `util.py` or `helper.py` module. But many developers dislike these generic names, especially Go developers who prefer more descriptive package names.

2. **Create a dedicated file**: Put it in `mess_up_things.py`. But then you need to write verbose import statements like:

```python
from mess_up_things import mess_up_things
```

This repetition feels unnecessary. Wouldn't it be nice if you could just do:

```python
import mess_up_things
mess_up_things()
```

## Exploring Solutions

If you're familiar with Python's magic methods, you might know that any object with a `__call__` method becomes callable. Since modules are instances of `types.ModuleType`, and they support magic methods like `__getattr__`, couldn't we just define a `__call__` method directly on a module?

Unfortunately, Python doesn't support this out of the box. There was actually [PEP 713](https://peps.python.org/pep-0713/) that proposed adding module-level `__call__` support, but it was rejected.

However, if you're familiar with Python's runtime, you'll find there's a trick to achieve this behavior:

```python
import sys

class MyModule(sys.modules[__name__].__class__):
    def __call__(self):
        print("Messing up things...")

sys.modules[__name__].__class__ = MyModule
```

This works by dynamically replacing the module's `__class__` at runtime. But who wants to copy-paste this boilerplate every time?

## Enter Cadule

To avoid repeating this boilerplate, I've encapsulated this simple mechanism into a package called **Cadule** (short for **Ca**llable **[Mo]dule** **Le**ss).

### Installation

```bash
pip install cadule
```

### Usage

Create a file called `mess_up_things.py`:

```python
import cadule

@cadule
def __call__():
    print("Messing up things...")
```

Then in your Python REPL or another script:

```python
>>> import mess_up_things
>>> mess_up_things()
Messing up things...
>>> callable(mess_up_things)
True
```

That's it! The entire `mess_up_things` module is now a callable object. When you call it, it executes the decorated `__call__` function.

You can also pass arguments and return values:

```python
import cadule

@cadule
def __call__(target):
    return f"Messing up {target}!"
```

```python
>>> import mess_up_things
>>> mess_up_things("the database")
'Messing up the database!'
```

## When to Use It

Cadule is particularly useful for:

- **Single-purpose modules**: When a module's main purpose is to expose one function
- **DSL and fluent interfaces**: Creating more natural-looking APIs
- **Scripts and utilities**: Making command-line tools more intuitive

---

Github: https://github.com/aisk/cadule
