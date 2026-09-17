import numpy as np

# Generate two vectors with 1,000,000 random floats (32-bit)
vector_a = np.random.random(1000000).astype(np.float32)
vector_b = np.random.random(1000000).astype(np.float32)

# Calculate sum of the vectors
vector_sum = vector_a + vector_b

# Write all three vectors to fixture.bin
with open('fixture.bin', 'wb') as f:
    # Write vector_a
    f.write(vector_a.tobytes())
    # Write vector_b
    f.write(vector_b.tobytes())
    # Write vector_sum
    f.write(vector_sum.tobytes())

print("Created fixture.bin with three vectors of 1,000,000 floats each")
