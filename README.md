# Multi-Layer Perceptron (MLP) From Scratch in C++

A robust implementation of a fully connected Neural Network (Multi-Layer Perceptron) built entirely from scratch in C++. 

This project was built to understand the low-level mathematics of Deep Learning—specifically **Backpropagation** and **Gradient Descent**—without relying on any high-level libraries like PyTorch, TensorFlow, or linear algebra frameworks (like Eigen).

## Project Overview

* **Language:** C++ (Standard Library only)
* **Architecture:** 784 Inputs $\to$ 64 Hidden Neurons $\to$ 10 Outputs
* **Dataset:** MNIST (Handwritten Digits)
* **Math:** Sigmoid Activation, Mean Squared Error (MSE) Loss, Stochastic Gradient Descent (SGD).

## The Benchmark: Custom C++ vs. PyTorch

To evaluate the correctness and efficiency of this implementation, I benchmarked it against an **architecturally equivalent** PyTorch model. 

### The Constraints
To make the comparison fair (since my C++ code uses scalar logic), the PyTorch baseline was intentionally constrained:
1.  **Single-Threaded:** (`torch.set_num_threads(1)`)
2.  **No Batching:** (Batch Size = 1, SGD behavior)
3.  **Identical Topology:** (784 $\to$ 64 $\to$ 10)

### Results (25,000 Training Samples)

| Metric | My C++ Implementation | PyTorch Baseline |
| :--- | :--- | :--- |
| **Accuracy** | **91.06%** | 89.34% |
| **Training Time** | 52.16s | **14.81s** |
| **Testing Time** | 4.23s | **0.24s** |

> **Note on Performance:** > Even with constraints, PyTorch is faster because it utilizes highly optimized C/C++ backends and contiguous memory arrays. My implementation uses `std::vector` objects
>  and `structs` for neurons, prioritizing **readability and object-oriented design** over cache coherence and vectorization.

## Project Structure

* `Neuron.cpp/h`: Defines a single neuron, handling weights, gradients, and activation functions (Sigmoid).
* `Net.cpp/h`: Manages layers, feed-forward logic, and backpropagation.
* `MnistCSV.h`: A custom CSV parser to load the MNIST dataset into `std::vector` containers.
* `main.cpp`: The entry point that orchestrates training, testing, and timing.
* `MLP.py`: The Python/PyTorch script used for benchmarking.

## How to Run

### Prerequisites
* A C++ Compiler (g++, clang, or MSVC)
* The `mnist_train.csv` file (placed in the root directory)

### Compilation
You can compile all source files together using g++:

```bash
g++ main.cpp Net.cpp Neuron.cpp -o main
```
#### Rerfrence
Huge thanks for Dave Miller providing a [brilliant tutorial](https://www.youtube.com/watch?v=sK9AbJ4P8ao&t=490s) that enhanced my understanding of this problem and coding this C++ without any additional libaries
