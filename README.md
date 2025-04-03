# Physics-Informed Neural Network (PINN) with PyTorch

This repository demonstrates how to build a simple Physics-Informed Neural Network (PINN) using PyTorch to solve a second-order ordinary differential equation (ODE):

u''(x) + π² u(x) = 0, with boundary conditions u(0) = 0 and u(1) = 0

The known analytical solution is:

u(x) = sin(πx)

## Overview

PINNs are neural networks trained not only on data points but also on known physical laws such as differential equations. This approach allows us to solve physical problems using deep learning, especially when data is limited.

In this example, we use a simple neural network and enforce the differential equation using PyTorch's automatic differentiation. We also enforce the boundary conditions and add one interior data point (u(0.5) = 1) to avoid the trivial solution u(x) = 0.

## Features

- Implements a fully connected neural network using PyTorch
- Uses automatic differentiation to compute derivatives
- Combines physics loss, boundary condition loss, and interior point loss
- Trains the model to approximate the sine function solution


## Requirements

- Python 3.7 or later
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook

## Installation

Install the required packages using pip:

```
pip install torch numpy matplotlib notebook
```

## Running the Project

1. Open a terminal and launch Jupyter Notebook:

```
jupyter notebook PINN.ipynb
```

2. Run the notebook to train the model and visualize the results.

## Output

The model learns to predict the function u(x) = sin(πx), which satisfies the given ODE and boundary conditions. A plot compares the model output with the exact analytical solution.

## License

This project is licensed under the [MIT License](LICENSE).
