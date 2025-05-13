from parsimonious.nodes import NodeVisitor
from parsimonious.grammar import Grammar
from pathlib import Path
import numpy as np
import logging
import sys

from AST import *  # Import your AST definitions

class TensorDSLVisitor(NodeVisitor):
    """Visitor for the parse tree"""
    
    def visit_program(self, node, visited_children):
        logging.debug("in program")
        block_lst = []
        for v in visited_children:
            if v[0] is not None:
                block_lst.append(v[0])
        return Block(block_lst)
    
    def visit_statement(self, node, visited_children):
        logging.debug("in statement")
        return visited_children[0][0]
    
    def visit_binding(self, node, visited_children):
        logging.debug("in binding")
        name = visited_children[1]
        value = visited_children[3]
        return Let(name, value)
    
    def visit_operation(self, node, visited_children):
        logging.debug("in operation")
        return visited_children[0]
    
    def visit_shard_op(self, node, visited_children):
        logging.debug("in shard_op")
        tensor = visited_children[4]
        dim = visited_children[8]
        device_group = visited_children[12]
        return Shard(tensor, dim, device_group)
    
    def visit_replicate_op(self, node, visited_children):
        logging.debug("in replicate_op")
        tensor = visited_children[4]
        device_group = visited_children[8]
        return Replicate(tensor, device_group)
    
    def visit_reduce_op(self, node, visited_children):
        logging.debug("in reduce_op")
        tensor = visited_children[4]
        dst = visited_children[8]
        device_group = visited_children[12]
        return Reduce(tensor, dst, device_group)
    
    def visit_gather_op(self, node, visited_children):
        logging.debug("in gather_op")
        tensor = visited_children[4]
        dim = visited_children[8]
        device_group = visited_children[12]
        return Gather(tensor, dim, device_group)
    
    def visit_visualize_op(self, node, visited_children):
        logging.debug("in visualize_op")
        tensor = visited_children[4]
        return Visualize(tensor)

    def visit_init_device(self, node, visited_children):
        logging.debug("in init_device")
        device_id = visited_children[2]
        return InitDevice(device_id)
    
    def visit_dim_list(self, node, visited_children):
        logging.debug("in dim_list")
        dims = []
        if visited_children[1]:
            for dim_comma in visited_children[1]:
                dims.append(dim_comma[0])
        dims.append(visited_children[2])
        return dims
    
    def visit_expr(self, node, visited_children):
        logging.debug("in expr")
        return visited_children[0]
    
    def visit_tensor_literal(self, node, visited_children):
        logging.debug("in tensor_literal")
        data = visited_children[2]
        device = visited_children[4]
        return TensorLiteral(np.array(data), device)
    
    def visit_tensor_data(self, node, visited_children):
        logging.debug("in tensor_data")
        rows = []
        if visited_children[1]:
            for row_comma in visited_children[1]:
                rows.append(row_comma[0])
        rows.append(visited_children[2])
        return rows
    
    def visit_tensor_row(self, node, visited_children):
        logging.debug("in tensor_row")
        values = []
        if visited_children[1]:
            for val_comma in visited_children[1]:
                values.append(val_comma[0])
        values.append(visited_children[2])
        return values
    
    def visit_device_group_literal(self, node, visited_children):
        logging.debug("in device_group_literal")
        data = visited_children[2]
        return DeviceGroupLiteral(np.array(data))
    
    def visit_device_group_data(self, node, visited_children):
        logging.debug("in device_group_data")
        rows = []
        if visited_children[1]:
            for row_comma in visited_children[1]:
                rows.append(row_comma[0])
        rows.append(visited_children[2])
        return rows
    
    def visit_device_row(self, node, visited_children):
        logging.debug("in device_row")
        devices = []
        if visited_children[1]:
            for dev_comma in visited_children[1]:
                devices.append(dev_comma[0])
        devices.append(visited_children[2])
        return devices
    
    def visit_variable(self, node, visited_children):
        logging.debug("in variable")
        return Variable(visited_children[0])
    
    def visit_number(self, node, visited_children):
        logging.debug("in number")
        return visited_children[0]
    
    def visit_float(self, node, visited_children):
        logging.debug("in float")
        return float(node.text.strip())
    
    def visit_integer(self, node, visited_children):
        logging.debug("in integer")
        return int(node.text.strip())
    
    def visit_name(self, node, visited_children):
        logging.debug("in name")
        return node.text.strip()
        
    def visit_emptyline(self, node, visited_children):
        logging.debug("in emptyline")
        return None
    
    def generic_visit(self, node, visited_children):
        return visited_children

def parse(file_name: str):
    grammar = Grammar(Path("grammar.peg").read_text())
    tree = grammar.parse(Path(file_name).read_text())
    
    visitor = TensorDSLVisitor()
    ast = visitor.visit(tree)
    return ast

# for testing
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: parse.py [file]")
        exit(1)

    grammar = Grammar(Path("grammar.peg").read_text())
    tree = grammar.parse(Path(sys.argv[1]).read_text())
    print(tree)
