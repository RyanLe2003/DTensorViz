from enum import Enum
from dataclasses import dataclass
import numpy as np

class Statement:
    """Statement base class"""
    pass

class Expr(Statement):
    """Expression base class"""
    pass

@dataclass
class Block:
    """A block of code containing multiple statements"""
    stmts: list[Statement]
    
    def __eq__(self, other):
        return type(self) is type(other) and self.stmts == other.stmts

# Core operations
@dataclass
class Shard(Statement):
    """Shard operation on a tensor"""
    tensor: Expr
    device_group: Expr
    
    def __eq__(self, other):
        return (type(self) is type(other) and 
                self.tensor == other.tensor and 
                self.device_group == other.device_group)

@dataclass
class Replicate(Statement):
    """Replicate operation on a tensor"""
    tensor: Expr
    device_group: Expr
    
    def __eq__(self, other):
        return (type(self) is type(other) and 
                self.tensor == other.tensor and 
                self.device_group == other.device_group)

@dataclass
class Reduce(Statement):
    """Reduce operation on a tensor"""
    tensor: Expr
    dst: Expr
    device_group: Expr
    
    def __eq__(self, other):
        return (type(self) is type(other) and 
                self.tensor == other.tensor and 
                self.dst == other.dst and 
                self.device_group == other.device_group)

@dataclass
class Gather(Statement):
    """Gather operation on a tensor"""
    tensor: Expr
    dim: list[int]
    device_group: Expr
    
    def __eq__(self, other):
        return (type(self) is type(other) and 
                self.tensor == other.tensor and 
                self.dim == other.dim and 
                self.device_group == other.device_group)

@dataclass
class Visualize(Statement):
    """Visualize the current tensor distribution"""
    tensor: Expr
    
    def __eq__(self, other):
        return type(self) is type(other) and self.tensor == other.tensor

@dataclass
class InitDevice(Statement):
    device: int

    def __eq__(self, other):
        return type(self) is type(other) and self.device == other.device



# Basic expressions
@dataclass
class TensorLiteral(Expr):
    """A tensor literal"""
    values: np.ndarray
    device: int
    
    def __eq__(self, other):
        return type(self) is type(other) and np.array_equal(self.values, other.values)

@dataclass
class DeviceGroupLiteral(Expr):
    """A device group literal"""
    devices: np.ndarray
    
    def __eq__(self, other):
        return type(self) is type(other) and self.devices == other.devices

@dataclass
class Variable(Expr):
    """A variable reference"""
    name: str
    
    def __eq__(self, other):
        return type(self) is type(other) and self.name == other.name

@dataclass
class Let(Statement):
    """Variable binding"""
    name: str
    value: Expr
    
    def __eq__(self, other):
        return (type(self) is type(other) and 
                self.name == other.name and 
                self.value == other.value)

