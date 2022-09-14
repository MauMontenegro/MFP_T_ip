import os
import sys
from pathlib import Path

from utils import utils

if __name__ == '__main__':
    only_graph = False
    args = utils.argParser(sys.argv[:])
    config = utils.getExpConfig(args.config)
    if only_graph:
        utils.generateGraph(config)
        #utils.generateGraphSeeds(config)
    else:
        solver = utils.createSolver(args.solver)
        path = Path.cwd() / "Experiments" / str(args.experiment)
        solver_ = solver(args.mode, args.load, path, config)
        solver_.solve()

