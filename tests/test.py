import numpy as np
from cucountlib import cucount

# Define input arrays
n = 1024
a = np.random.rand(n).astype(np.float32)
b = np.random.rand(n).astype(np.float32)
c = np.zeros_like(a)

# Call the CUDA function
cucount.vector_add(a, b, c)

# Verify the result
assert np.allclose(c, a + b)
print("Vector addition successful!")