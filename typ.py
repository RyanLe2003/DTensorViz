from AST import *
from enum import Enum
from typing import Dict

class Type(Enum):
    TENSOR = "Tensor"
    DEVICE_GROUP = "DeviceGroup"
    INT = "Int"
    VOID = "Void"

class TypeChecker:
    def __init__(self):
        self.var_types: Dict[str, Type] = {}

    def check_block(self, block):
        for stmt in block.stmts:
            self.check_stmt(stmt)

    def check_stmt(self, stmt):
        if isinstance(stmt, Let):
            t = self.check_expr(stmt.value)
            self.var_types[stmt.name.name] = t
        elif isinstance(stmt, Shard):
            if self.check_expr(stmt.tensor) != Type.TENSOR:
                raise TypeError("Shard: tensor must be Tensor")
            if self.check_expr(stmt.device_group) != Type.DEVICE_GROUP:
                raise TypeError("Shard: device_group must be DeviceGroup")
        elif isinstance(stmt, Replicate):
            if self.check_expr(stmt.tensor) != Type.TENSOR:
                raise TypeError("Replicate: tensor must be Tensor")
            if self.check_expr(stmt.device_group) != Type.DEVICE_GROUP:
                raise TypeError("Replicate: device_group must be DeviceGroup")
        elif isinstance(stmt, Reduce):
            if self.check_expr(stmt.tensor) != Type.TENSOR:
                raise TypeError("Reduce: tensor must be Tensor")
        elif isinstance(stmt, Gather):
            if self.check_expr(stmt.tensor) != Type.TENSOR:
                raise TypeError("Gather: tensor must be Tensor")
        elif isinstance(stmt, Visualize):
            if self.check_expr(stmt.tensor) != Type.TENSOR:
                raise TypeError("Visualize: tensor must be Tensor")
        elif isinstance(stmt, InitDevice):
            if not isinstance(stmt.device, int):
                raise TypeError("InitDevice: device must be Int")
        elif isinstance(stmt, Matmul):
            if self.check_expr(stmt.tensor_one) != Type.TENSOR or self.check_expr(stmt.tensor_two) != Type.TENSOR:
                raise TypeError("Matmul: both operands must be Tensor")
        elif isinstance(stmt, Relu):
            if self.check_expr(stmt.tensor) != Type.TENSOR:
                raise TypeError("Relu: tensor must be Tensor")
        else:
            raise TypeError(f"Unknown statement: {stmt}")

    def check_expr(self, expr):
        if isinstance(expr, TensorLiteral):
            return Type.TENSOR
        elif isinstance(expr, DeviceGroupLiteral):
            return Type.DEVICE_GROUP
        elif isinstance(expr, Variable):
            if expr.name not in self.var_types:
                raise TypeError(f"Variable {expr.name} not defined")
            return self.var_types[expr.name]
        elif isinstance(expr, Matmul):
            if self.check_expr(expr.tensor_one) != Type.TENSOR or self.check_expr(expr.tensor_two) != Type.TENSOR:
                raise TypeError("Matmul: both operands must be Tensor")
            return Type.TENSOR
        elif isinstance(expr, Relu):
            if self.check_expr(expr.tensor) != Type.TENSOR:
                raise TypeError("Relu: tensor must be Tensor")
            return Type.TENSOR
        else:
            raise TypeError(f"Unknown expr: {expr}")
