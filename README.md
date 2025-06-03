# DTensorViz
DTensorViz is a domain specific language designed to help users learn and visualize operations on tensors, particularly operations that could be distributed in the context of distributed neural networks. Currently, DTensorViz supports only 2-dimensional tensors.

## Installation
Before running this project, you need to install the following Python packages:
-numpy
-matplotlib
-parsimonious

You can install all required dependencies with:
```
pip install numpy matplotlib parsimonious
```

Alternatively, you can install all dependencies at once with:
```
pip install -r requirements.txt
```


## Initializing a device
Distributed training (operations) typically require at least two devices to parallelize computation. DTensorViz circumvents this by simulating devices as no true parallelization is needed for visualization, and DTensorViz is not designed for training. To create a device, call `init_dev`. `init_dev` requires you to provide a unique identifier (int) as a parameter.

```
init_dev(1);
```

## Creating a tensor
To create a tensor, instantiate a `tensor` object. The `tensor` object requires you to provide a 2-d list as a parameter.
```
let list = [[1, 2], [3, 4]];
let tensor_name = tensor(tensor=list);
```

## Creating a device group
To create a device group, instantiate a `devices` object. The `devices` object requires you to provide a 2-d list as a parameter.
The devices included in the list must be devices that have already been initialized utilizing `init_dev`.

```
let device_group_name = devices([[1, 2]]);
```

## Performing parllel operations on a tensor

DTensorViz currently supports the following parallelization operations:

### Shard
Splits a tensor across a group of devices, distributing different parts of the tensor to each device. The provided tensor can not already be sharded or replicated. To perform a shard operation use `shard`.
```
shard(tensor=my_tensor, device_group=dg);
```

### Replicate
Copies the entire tensor to each device in the specified device group. The provided tensor can not already be sharded or replicated. To perform a replicate operation use `replicate`.
```
replicate(tensor=my_tensor, device_group=dg);
```

### Gather
Collects sharded tensor pieces from multiple devices and assembles them onto a single destination device. The provided tensor must be sharded. To perform a gather operation use `gather`.

```
gather(tensor=my_tensor, dst=1);
```

### Reduce
Aggregates (e.g., sums or averages) tensor data from multiple devices onto a destination device. The provided tensor must be replicated. To perform a reduce operation use `reduce`.
```
reduce(tensor=my_tensor, dst=1);
```

## Performing algebraic operations on a tensor

DTensorViz currently supports the following algebraic operations:

### Matrix Multiplication

Multiplies two tensors using standard matrix multiplication rules and returns a new tensor. To perform matrix multiplication use `matmul`.

```
let new_tensor = matmul(tensor_one, tensor_two);
```

### RELU
Performs a the relu activation function and returns a new tensor. To perform relu use `relu`.
```
let new_tensor = relu(tensor_one);
```

#### Note: Matrix multiplication on distributed tensors

When performing matrix multiplication on distributed tensors, the parallelization (sharding or replication) of the result depends on the parallelization of the input tensors:


| Tensor One  | Tensor Two | Result |
| ------------- | ------------- | ------------- |
| Replicated  | Replicated  | Replicated |
| Shard  | Shard  | Replicated |
| Shard | Replicated/None | Shard |
| Replicated/None | Shard | Shard |
|None | None | None |


## Visualizing a tensor

Displays the current state of a tensor for inspection and learning. To visualize a tensor use `visualize`.

```
visualize(tensor=my_tensor);
```

## Getting Started Examples

### Shard and Combine

The example below demonstrates the sharding of a tensor along its columns and gathering it back together. The visualization calls are placed to show each step.
```
init_dev(1);
init_dev(2);

let tensor_col_ex = tensor([[1, 2], [3, 4]], dev=1);
visualize(tensor=tensor_col_ex);

let dg_1 = devices([[1, 2]]);
shard(tensor=tensor_col_ex, device_group=dg_1);
visualize(tensor=tensor_col_ex);

gather(tensor=tensor_col_ex, dst=1);
visualize(tensor=tensor_col_ex);
```

The example below demonstrates the sharding of a tensor along its rows and gathering it back together. The visualization calls are placed to show each step.
```
init_dev(1);
init_dev(2);

let tensor_row_ex = tensor([[1, 2], [3, 4]], dev=1);
visualize(tensor=tensor_row_ex);

let dg_1 = devices([[1], [2]]);
shard(tensor=tensor_row_ex, device_group=dg_1);
visualize(tensor=tensor_row_ex);

gather(tensor=tensor_row_ex, dst=1);
visualize(tensor=tensor_col_ex);
```

### Replicate and Reduce

The example below demonstrates the replicating of a tensor reducing it back together. The visualization calls are placed to show each step.
```
init_dev(1);
init_dev(2);

let tensor_rep_ex = tensor([[1, 2], [3, 4]], dev=1);
visualize(tensor=tensor_rep_ex);

let dg_1 = devices([[1, 2]]);
replicate(tensor=tensor_rep_ex, device_group=dg_1);
visualize(tensor=tensor_rep_ex);

reduce(tensor=tensor_rep_ex, dst=2);
visualize(tensor=tensor_rep_ex);
```

## Advanced Examples

### Data Parallelism

The example below demonstrates a data parallel distributed training setup visualized.
```
init_dev(1);
init_dev(2);

let tensor_one = tensor([[1, 2], [3, 4]], dev=1);
visualize(tensor=tensor_one);

let tensor_two = tensor([[2]], dev=1);
visualize(tensor=tensor_two);

let dg_1 = devices([[1, 2]]);
shard(tensor=tensor_one, device_group=dg_1);
visualize(tensor=tensor_one);

let dg_2 = devices([[1, 2]]);
replicate(tensor=tensor_two, device_group=dg_2);
visualize(tensor=tensor_two);

let split_matmul = matmul(tensor_one, tensor_two);
visualize(tensor=split_matmul);

gather(tensor=split_matmul, dst=1);
visualize(tensor=split_matmul);
```

### Reduction Parallelism
The example below demonstrates a reduction parallel distributed training setup visualized.
```
init_dev(1);
init_dev(2);

let tensor_one = tensor([[1, 2], [3, 4]], dev=1);
visualize(tensor=tensor_one);
let tensor_two = tensor([[2, 0], [0, 2]], dev=1);
visualize(tensor=tensor_two);

let dg_1 = devices([[1, 2]]);
shard(tensor=tensor_one, device_group=dg_1);
visualize(tensor=tensor_one);

let dg_2 = devices([[1], [2]]);
shard(tensor=tensor_two, device_group=dg_2);
visualize(tensor=tensor_two);

let split_matmul = matmul(tensor_one, tensor_two);
visualize(tensor=split_matmul);

reduce(tensor=split_matmul, dst=1);
visualize(tensor=split_matmul);
```

### 2 Layer MLP
The example below demonstrates a 2layerMLP distributed training setup visualized.
```
init_dev(0);
init_dev(1);

let input = tensor([[2, -3, 0, 5, 2, 4, 0, 2]], dev=0);
visualize(tensor=input);

let weight_1 = tensor([[5, -1, 4, 1], [2, 3, 7, 2], [-4, -6, 2, 9], [9, -1, 3, 1], [2, 3, 4, 5], [-9, -3, 1, 4], [1, 2, 3, 4], [0, 9, 9, 9]], dev=0);
visualize(tensor=weight_1);

let dev_row = devices([[0, 1]]);
let dev_col = devices([[0], [1]]);

shard(tensor=input, device_group=dev_row);
visualize(tensor=input);

shard(tensor=weight_1, device_group=dev_col);
visualize(tensor=weight_1);

let layer_1 = matmul(input, weight_1);
reduce(tensor=layer_1, dst=0);
replicate(tensor=layer_1, device_group=dev_row);

let relu1 = relu(layer_1);

visualize(tensor=relu1);

let weight_2 = tensor([[2, 1], [5, 6], [0, 1], [-3, 4]], dev=0);
visualize(tensor=weight_2);
shard(tensor=weight_2, device_group=dev_row);

visualize(tensor=weight_2);

let layer_2 = matmul(relu1, weight_2);

visualize(tensor=layer_2);

let relu2 = relu(layer_2);

visualize(tensor=relu2);

gather(tensor=relu2, dst=0);

visualize(tensor=relu2);
```

