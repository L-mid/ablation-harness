import sys

import matplotlib
import numpy
import omegaconf

import ablation_harness

print("exe:", sys.executable)
print("py:", sys.version.splitlines()[0])
print("pkg:", ablation_harness.__file__)
print("numpy:", numpy.__version__)
print("omegaconf:", omegaconf.__version__)
print("matplotlib:", matplotlib.__version__)
