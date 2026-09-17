# Micrograd Autograd: Lightweight Scalar Automatic Differentiation Engine

A clean, light, scalar-valued Automatic Differentiation (Autograd) engine and Neural Network framework implemented completely from scratch in pure Python.

---

## **Key Highlights**
* **Dynamic Computational Graph:** Dynamically constructs Directed Acyclic Graphs (DAGs) on every forward operation pass.
* **Reverse-Mode Backpropagation:** Performs complete topological ordering and computes exact partial derivatives via the multivariable chain rule.
* **Neural Network Modules:** Includes custom `Neuron`, `Layer`, and `MLP` (Multi-Layer Perceptron) building blocks supporting activations like $\tanh$ and $\text{ReLU}$.
* **Zero External Dependencies:** Built entirely with standard library tools (`math`, `random`) without relying on PyTorch, NumPy, or TensorFlow.

---

## **Repository Architecture**

```text
├── engine.py       # Core Value class, Autograd Engine, and MLP framework
└── README.md       # Project documentation
