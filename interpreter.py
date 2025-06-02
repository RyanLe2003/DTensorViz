from AST import *
import numpy as np
import logging
from DistributedTensor import *
import matplotlib.pyplot as plt

all_devices = set()

def interpret_expr(expr: Expr, bindings: dict):
    """Interpret an expression. Bindings is a dictionary that maps
    variable names to values.
    
    Returns the value the expression evaluates to.
    """
    logging.debug("In interpret_expr")
    
    match expr:
        case str():
            logging.debug("In string")  # this is so hacky lol
            return bindings[expr]
        case Variable(name=name):
            logging.debug("In variable")
            return bindings[name]
        
        case TensorLiteral(values=values,device=device):
            logging.debug("In tensor literal")
            return DistributedTensor(values, device)
        
        case DeviceGroupLiteral(devices=devices):
            logging.debug("In device group literal")
            return devices
        case _:
            return interpret_stmt(expr, bindings)


def interpret_stmt(stmt: Statement, bindings: dict):
    """Interpret a statement.
    
    Returns the result of the statement, if any.
    """
    logging.debug("In interpret_stmt")
    
    match stmt:
        case Let(name=name, value=expr):
            logging.debug("In let")
            value = interpret_expr(expr, bindings)
            bindings[name] = value
            return None
        
        case Shard(tensor=tensor, device_group=device_group):
            logging.debug("In shard")
            tensor_val = interpret_expr(tensor, bindings)
            device_group_val = interpret_expr(device_group, bindings)
            
            result = manual_shard(tensor_val, device_group_val)
            return result
        
        case Replicate(tensor=tensor, device_group=device_group):
            logging.debug("In replicate")
            tensor_val = interpret_expr(tensor, bindings)
            device_group_val = interpret_expr(device_group, bindings)
            
            result = manual_replicate(tensor_val, device_group_val)
            return result
        
        case Reduce(tensor=tensor, dst=dst):
            logging.debug("In reduce")
            tensor_val = interpret_expr(tensor, bindings)
            
            result = manual_reduce(tensor_val, dst)
            return result
        
        case Gather(tensor=tensor, dst=dst):
            logging.debug("In gather")
            tensor_val = interpret_expr(tensor, bindings)
            result = manual_gather(tensor_val, dst)
            return result
        
        case Visualize(tensor=tensor):
            logging.debug("In visualize")
            tensor_val = interpret_expr(tensor, bindings)

            device_group = tensor_val.cur_dev_group
            device_slices = extract_device_slices(device_group, tensor_val.full_tensor.shape)
            
            visualize_tensor(tensor, tensor_val, device_slices)
            return None

        case InitDevice(device=device):
            logging.debug("in init device")
            all_devices.add(device)
            return
        
        case Matmul(tensor_one=tensor_one, tensor_two=tensor_two):
            logging.debug("In matmul")
            tensor_one = interpret_expr(tensor_one, bindings)
            tensor_two = interpret_expr(tensor_two, bindings)
            result = manual_matmul(tensor_one, tensor_two)
            return result
        
        case Relu(tensor=tensor):
            logging.debug("In relu")
            tensor = interpret_expr(tensor, bindings)
            result = manual_relu(tensor)

            return result


def interpret_block(block: Block, bindings: dict):
    """Interpret each statement in the block.
    
    Returns the result of the last statement, if any.
    """
    logging.debug("In interpret_block")
    
    result = None
    for stmt in block.stmts:
        result = interpret_stmt(stmt, bindings)
    
    return result

def manual_shard(tensor, device_group):
    tensor.shard(device_group, all_devices)
    
def manual_replicate(tensor, device_group):
    tensor.replicate(device_group, all_devices)

def manual_reduce(tensor, dst):
    tensor.reduce(dst, all_devices)

def manual_gather(tensor, dst):
    tensor.gather(dst, all_devices)

def manual_matmul(tensor_one, tensor_two):
    # check if matching devices
    flattened_list_one = [item for sublist in tensor_one.cur_dev_group for item in sublist]
    flattened_list_one.sort()
    flattened_list_two = [item for sublist in tensor_two.cur_dev_group for item in sublist]
    flattened_list_two.sort()

    if flattened_list_one != flattened_list_two:
        raise RuntimeError("Devices on the two tensors are not the same")

    new_map = {}
    for device in flattened_list_one:
        cur_tens_1 = tensor_one.device_map[device]
        cur_tens_2 = tensor_two.device_map[device]

        if cur_tens_1.shape[1] != cur_tens_2.shape[0]:
            raise ValueError(f"Shapes {cur_tens_1.shape} and {cur_tens_2.shape} not compatible for matrix multiplication")

        res = np.matmul(cur_tens_1, cur_tens_2)
        new_map[device] = res

    # need to update shape (only really matters for partition though): device_group
    new_tensor = DistributedTensor(res)
    new_tensor.device_map = new_map

    # Inherit device group structure from first tensor (This is a design choice that I might change)
    new_tensor.cur_dev_group = tensor_one.cur_dev_group

    new_tensor.is_shard = False
    new_tensor.is_replicated = False
    if (tensor_one.is_replicated and tensor_two.is_replicated):
        new_tensor.is_replicated = True
    elif (tensor_one.is_shard and tensor_two.is_shard):
        new_tensor.is_replicated = True 
    elif (tensor_one.is_shard and not tensor_two.is_shard):
        new_tensor.is_shard = True
        new_tensor.cur_dev_group = tensor_one.cur_dev_group
        new_tensor.full_tensor = tensor_one.full_tensor  # a little hacky for now
    elif (not tensor_one.is_shard and tensor_two.is_shard):
        new_tensor.is_shard = True
        new_tensor.cur_dev_group = tensor_two.cur_dev_group
        new_tensor.full_tensor = tensor_two.full_tensor  # a little hacky for now
    return new_tensor

def manual_relu(distributed_tensor):
    tensor = distributed_tensor.full_tensor
    rows = len(tensor)
    cols = len(tensor[0])
    result = []
    
    for i in range(rows):
        row = []
        for j in range(cols):
            if tensor[i][j] > 0:
                row.append(tensor[i][j])
            else:
                row.append(0)  # Use integer 0 instead of 0.0 for consistency
        result.append(row)

    # Create new DistributedTensor with ReLU applied
    result_array = np.array(result)
    new_tensor = DistributedTensor(result_array)


    new_device_map = {}
    for device, local_tensor in distributed_tensor.device_map.items():
        local_rows = len(local_tensor)
        local_cols = len(local_tensor[0])
        local_result = []
        
        for i in range(local_rows):
            row = []
            for j in range(local_cols):
                if local_tensor[i][j] > 0:
                    row.append(local_tensor[i][j])
                else:
                    row.append(0)
            local_result.append(row)
        
        new_device_map[device] = np.array(local_result)
    
    # Preserve the distribution properties
    new_tensor.device_map = new_device_map
    new_tensor.cur_dev_group = distributed_tensor.cur_dev_group
    new_tensor.is_shard = distributed_tensor.is_shard
    new_tensor.is_replicated = distributed_tensor.is_replicated
    
    return new_tensor

def extract_device_slices(device_group, full_tensor_shape):
    if isinstance(device_group, list):
        device_group = np.array(device_group)
    w_t, h_t = full_tensor_shape
    w_d, h_d = device_group.shape
    row_shard = w_t // w_d
    col_shard = h_t // h_d

    device_slices = {}
    for i in range(w_d):
        for j in range(h_d):
            device = device_group[i, j]
            row_start = i * row_shard
            row_end = (i + 1) * row_shard
            col_start = j * col_shard
            col_end = (j + 1) * col_shard
            device_slices[device] = (slice(row_start, row_end), slice(col_start, col_end))
    return device_slices

def visualize_tensor(tensor_name, tensor, device_slices=None):
    full_tensor = tensor.full_tensor
    full_shape = full_tensor.shape
    device_map = tensor.device_map
    num_devices = len(device_map)
    is_replicated = tensor.is_replicated

    # Gather all data for global color normalization
    all_chunks = [chunk for chunk in device_map.values()]
    global_min = min(chunk.min() for chunk in all_chunks)
    global_max = max(chunk.max() for chunk in all_chunks)

    fig, axes = plt.subplots(1, num_devices, figsize=(5 * num_devices, 5), squeeze=False)
    fig.suptitle(f"Tensor Distribution: {tensor_name}", fontsize=22)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='lightgray')

    for ax, (device, chunk) in zip(axes[0], device_map.items()):
        if is_replicated:
            display_array = chunk
        else:
            display_array = np.full(full_shape, np.nan, dtype=float)
            if device_slices is None:
                raise ValueError("device_slices must be provided for partitioned tensors.")
            indices = device_slices[device]

            display_array[indices] = chunk

        im = ax.imshow(display_array, cmap=cmap, vmin=global_min, vmax=global_max)
        ax.set_title(f"Device {device}", fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

        for (i, j), value in np.ndenumerate(display_array):
            if not np.isnan(value):
                ax.text(j, i, f"{int(value)}", ha="center", va="center", color="black", fontsize=14)
    
    print("\n" + "=" * 40)
    print(f"## {tensor_name} ##".center(40))
    print("=" * 40)
    
    for key, val in tensor.device_map.items():
        print(f"\nDevice {key}:".ljust(40, "-"))
        
        if hasattr(val, 'tolist'):
            data = val.tolist()
        else:
            data = val
            
        for row in data:
            print(str(row).center(40))
    
    print("=" * 40 + "\n")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()