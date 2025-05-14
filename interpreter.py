from AST import *
import numpy as np
import logging
from DistributedTensor import *

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
            
            visualize_tensor(tensor, tensor_val)
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
    new_tensor = DistributedTensor()
    new_tensor.device_map = new_map

    # Inherit device group structure from first tensor (This is a design choice that I might change)
    new_tensor.cur_dev_group = tensor_one.cur_dev_group

    new_tensor.is_shard = False
    new_tensor.is_replicated = False
    if (tensor_one.is_shard and tensor_two.is_shard):
        new_tensor.is_replicated = True
    elif (tensor_one.is_shard or tensor_two.is_shard):
        new_tensor.is_replicated = True
        new_tensor.is_shard = True
    
    if (tensor_one.is_replicated and tensor_two.is_replicated):  # this should always be true if at least one is
        new_tensor.is_replicated = True

    return new_tensor

def visualize_tensor(tensor_name, tensor):
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
