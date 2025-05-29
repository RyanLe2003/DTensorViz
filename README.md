# DTensorViz
DTensorViz is a domain specific language used to visualize distributed tensors in the context of distributed neural network training. Currently DTensorViz supports only 2-dimensional tensors.

## Initializing a device
Distributed setups require at least two devices. DTensorViz circumvents this by simulating devices as no true parallelization is needed for visualization. However, it is still requires initialization of simulation devices. To create a device, call `init_dev`. `init_dev` requires you to provide a unique identifier (int) as a parameter.

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

```
shard(tensor=my_tensor, device_group=dg);
```

### Replicate

```
replicate(tensor=my_tensor, device_group=dg);
```

### Gather

```
gather(tensor=my_tensor, dst=1);
```

### Reduce
```
reduce(tensor=my_tensor, dst=1);
```

## Performing algebraic operations on a tensor

DTensorViz currently supports the following algebraic operations:

### Matrix Multiplication

```
let new_tensor = matmul(tensor_one, tensor_two);
```

## Visualizing a tensor

To visualize a tensor use `visualize`

```
visualize(tensor=my_tensor);
```

## Example Programs

