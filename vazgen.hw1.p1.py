import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def plot(variances, trials=100):
  dd = defaultdict(list)  # Dictionary to store information as we iterate
  for _ in range(trials):
    for var in variances:
      X = np.random.uniform(-20, 80, 100)
      # define a function using lambda notation f(x) = 0.2x + 15
      f = lambda x: 0.2 * x + 15
      # add gaussian noise, y* = f(x) + N(0, sigma^2)
      Y = [np.random.normal(f(x), np.sqrt(var)) for x in X]
      def run_regression(data_list, response_list):
        X = np.array([[1, x] for x in data_list])
        Y = np.array(response_list)
        [b0, b1] = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        return b0, b1
      b0, b1 = run_regression(X, Y)
      # Calculate error for the given trial
      dd[var].append((b0 - 15) ** 2 + (b1 - 0.2) ** 2)
  var_err = []
  # Calculate means and standard deviations, for each variance
  for var, errors in dd.items():
    var_err.append((var, np.mean(errors), np.std(errors)))  

  # Plot
  x, y, yerr = zip(*sorted(var_err))
  plt.errorbar(x, y, yerr=yerr)
  plt.xlabel('true error variance (sigma star)')
  plt.ylabel('beta error (avaraged over 1000 trials)')
  plt.savefig("vazgen.hw1.p1.png", dpi=320, bbox_inches='tight')

plot([0.1, 0.33, 1., 3.3, 10.], trials=1000)

plt.show()
