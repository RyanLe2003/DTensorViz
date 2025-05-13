from AST import *
import numpy as np
import logging

all_devices = set()

class DistributedTensor:
    """Class to simulate a distributed tensor across multiple devices""" 
    
    def __init__(self, tensor, device):
        """
        Initialize a distributed tensor
        
        Args:
            tensor: The full tensor data (numpy array)
            device_map: Dictionary mapping device IDs to tensor chunks
                        If None, the tensor is considered to be on a single device (0)
        """
        self.full_tensor = tensor
        self.device_map = {device: self.full_tensor}
        self.is_shard = False
    
    def shard(self, dim, device_group):
        if self.is_shard:
            raise RuntimeError("You cannot shard an already sharded tensor, please use a collective first!")
        
        for row in device_group:
            for device in row:
                if device not in all_devices:
                    raise RuntimeError("Device in device group not an initialized device")
        
        w_t, h_t = self.full_tensor.shape
        w_d, h_d = device_group.shape
            
        area_dg = w_d * h_d
        area_tensor = w_t * h_t
        if  area_dg > area_tensor:
            raise RuntimeError("More Devices than possible shards")
        
        repeat_factors = (w_t // w_d, h_t // h_d)
    
        mapped_array = np.tile(device_group, repeat_factors)

        unique_values = np.unique(mapped_array)
    
        # Create a dictionary to store grouped elements
        grouped_elements = {}
        
        # For each unique value, extract elements from large array
        for value in unique_values:
            # Use np.where to find matching positions
            matches = np.where(mapped_array == value)
            
            # Extract elements from large array at these positions
            grouped_elements[int(value)] = self.full_tensor[matches]
        
        self.device_map = grouped_elements
    
    def __repr__(self):
        return f"DistributedTensor(data={self.full_tensor}, devices={list(self.device_map.keys())})"

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
        
        case Shard(tensor=tensor, dim=dim, device_group=device_group):
            logging.debug("In shard")
            tensor_val = interpret_expr(tensor, bindings)
            device_group_val = interpret_expr(device_group, bindings)
            
            result = manual_shard(tensor_val, dim, device_group_val)
            return result
        
        case Replicate(tensor=tensor, device_group=device_group):
            logging.debug("In replicate")
            tensor_val = interpret_expr(tensor, bindings)
            device_group_val = interpret_expr(device_group, bindings)
            
            result = manual_replicate(tensor_val, device_group_val)
            return result
        
        case Reduce(tensor=tensor, dst=dst, device_group=device_group):
            logging.debug("In reduce")
            tensor_val = interpret_expr(tensor, bindings)
            dst_val = interpret_expr(dst, bindings)
            device_group_val = interpret_expr(device_group, bindings)
            
            result = manual_reduce(tensor_val, dst_val, device_group_val)
            return result
        
        case Gather(tensor=tensor, dim=dim, device_group=device_group):
            logging.debug("In gather")
            tensor_val = interpret_expr(tensor, bindings)
            device_group_val = interpret_expr(device_group, bindings)
            
            result = manual_gather(tensor_val, dim, device_group_val)
            return result
        
        case Visualize(tensor=tensor):
            logging.debug("In visualize")
            tensor_val = interpret_expr(tensor, bindings)
            
            visualize_tensor(tensor_val)
            return None

        case InitDevice(device=device):
            logging.debug("in init device")
            all_devices.add(device)
            return



def interpret_block(block: Block, bindings: dict):
    """Interpret each statement in the block.
    
    Returns the result of the last statement, if any.
    """
    logging.debug("In interpret_block")
    
    result = None
    for stmt in block.stmts:
        result = interpret_stmt(stmt, bindings)
    
    return result

# Implement these functions based on the raw PyTorch operations
def manual_shard(tensor, dim, device_group):
    tensor.shard(dim, device_group)
    

def manual_replicate(tensor, device_group):
    # Implementation using torch.distributed primitives
    pass

def manual_reduce(tensor, dst, device_group):
    # Implementation using torch.distributed primitives
    pass

def manual_gather(tensor, dim, device_group):
    # Implementation using torch.distributed primitives
    pass

def visualize_tensor(tensor):
    print(tensor.device_map)
