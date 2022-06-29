import numpy as np

# monte carlo method to estimate the value of pi.
n_samples = 1000000
radius = 1
# rand returns samples from uniform distribution between [-1, 1]
x_samples = np.random.uniform(-1, 1, n_samples)
y_samples = np.random.uniform(-1, 1, n_samples)
in_circle = np.sqrt(np.square(x_samples) + np.square(y_samples)) <= 1
p_incircle = np.sum(in_circle) / n_samples
area_circle = p_incircle * 4
pi = area_circle
print(pi)
