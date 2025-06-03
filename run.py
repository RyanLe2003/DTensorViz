import numpy as np
import sys
from parse import parse
from interpreter import interpret_block
from typ import TypeChecker

import logging

# logging.basicConfig(level=logging.WARNING)
# logging.basicConfig(level=logging.DEBUG)
def main():
    if len(sys.argv) != 2:
        print("Usage: run.py [file]")
        exit(1)

    ast = parse(sys.argv[1])

    logging.debug("DONE WITH PARSING")

    # type_block(ast, bindings, declarations)
    TypeChecker().check_block(ast)

    logging.debug("DONE WITH TYPE CHECKING")
    bindings = {}

    interpret_block(ast, bindings)

if __name__ == "__main__":
    main()