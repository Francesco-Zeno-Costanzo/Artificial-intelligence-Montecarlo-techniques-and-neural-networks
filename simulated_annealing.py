import numpy as np
import matplotlib.pyplot as plt


np.random.seed(69420)


def simulated_annealing(func, bounds, step=0.5, tol=1e-8, temp=10, cooling_func=None):
    '''
    Find the minimum of a function using Simulated Annealing.

    Parameters
    ----------
    func : callable
        The function to minimize.
    bounds : list of tuples
        The bounds of the search space.
    step : float, optional, default 0.5
        The metropolis step to take in the search space.
    tol : float, optional, default 1e-8
        The required tollerance for the minimum.
    temp : float, optional, default 10
        The initial temperature.
    cooling_func : None or callable, optional, default None
        The cooling function to use. If None, the default
        (T_{i+1} = rate*T) cooling function is used.
        If a custom function is passed, it must take the
        current temperature and the iteration number as arguments.
    
    Returns
    -------
    global_best_x : list
        The best point found i.e. the minimum.
    global_best_value : float
        The value of the function at the minimum.
    '''
    
    D          = len(bounds)                                                    # Number of dimensions
    best_x     = [np.random.uniform(bound[0], bound[1]) for bound in bounds]    # Initial point
    best_value = func(*best_x)                                                  # Initial value
    count      = 0                                                              # Iteration counter

    # Track the global best solution
    global_best_x     = best_x.copy()
    global_best_value = best_value

    if cooling_func is None:
        cooling_func = lambda T, i: T * 0.99

    while True:
        count += 1 # Increase iteration counter
       
        # Sample a new point with uniform proposal distribution
        new_x = best_x + step*np.random.uniform(-1, 1, size=D)

        # Clip the point to ensure it is within the bounds
        new_x = [np.clip(x, bound[0], bound[1]) for x, bound in zip(new_x, bounds)]

        # Compute the new value and the "energy" difference
        new_value = func(*new_x)
        delta_E   = new_value - best_value

        # Metropolis acceptance criterion
        if np.log(np.random.rand()) < -delta_E / temp :
            best_value = new_value
            best_x     = new_x

            # Update the global best if improved
            if new_value < global_best_value:
                global_best_x     = new_x.copy()
                global_best_value = new_value
        
        # Decrease the temperature
        temp = cooling_func(temp, count)
        
        if global_best_value < tol or temp < tol:
            break
    
    return global_best_x, global_best_value, count

#================= Cooling functions =================

def exponential_cooling(T, i, cooling_rate=0.999):
    return T * cooling_rate

def logarithmic_cooling(T, i, alpha=0.001):
    return T / (1 + alpha * np.log(1 + i))

def inverse_cooling(T, i, alpha=0.001):
    return T / (1 + alpha * i)

#================= Function to minimize =================

def F(x, y):
    '''
    Ackley's function
    '''
    a, b, c = 20, 0.2, 2 * np.pi
    return -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(c*x) + np.cos(c*y))) + a + np.exp(1)

#================= Run the algorithm =================

bounds = [(-5.0, 5.0) for _ in range(2)]
best_x, best_value, nc = simulated_annealing(F, bounds, cooling_func=logarithmic_cooling, step=0.5)

print(f"Minimum in: x={best_x}, value={best_value:.5f}, with {nc} iterations")

#================= Plotting the results =================

X = np.linspace(-5, 5, 500)
Y = np.linspace(-5, 5, 500)
X, Y = np.meshgrid(X, Y)


plt.figure(figsize=(8, 6))
c=plt.contour(X, Y, F(X, Y), levels=50, cmap='plasma')
plt.colorbar(c)
plt.plot(best_x[0], best_x[1], "go", label="Found minimum")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
