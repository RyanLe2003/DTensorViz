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
        grouped_elements = {}
        
        for value in unique_values:
            positions = np.where(mapped_array == value)
        
            rows, cols = positions
            unique_rows = np.unique(rows)
            
            structured_result = []
            for row in unique_rows:
                row_indices = np.where(rows == row)[0]
                row_cols = cols[row_indices]
                sorted_indices = np.argsort(row_cols)
                row_cols = row_cols[sorted_indices]
                row_indices = row_indices[sorted_indices]
                
                row_values = [self.full_tensor[rows[i], cols[i]] for i in row_indices]
                structured_result.append(row_values)
            
            grouped_elements[int(value)] = np.array(structured_result)
        
        self.device_map = grouped_elements
        self.is_shard = True
    
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

def manual_shard(tensor, device_group):
    tensor.shard(tensor, device_group)
    

def manual_replicate(tensor, device_group):
    pass

def manual_reduce(tensor, dst, device_group):
    pass

def manual_gather(tensor, dim, device_group):
    pass

def visualize_tensor(tensor):
    for key, val in tensor.device_map.items():
        print("---------------------------")
        print(f"Device {key}")
        print(val.tolist())
    print("---------------------------")
