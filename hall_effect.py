import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Data for Hall effect study
# Current in mA and Hall Voltage in mV
current = np.array([0.051, 0.103, 0.152, 0.205, 0.249, 0.304, 0.358, 0.410, 0.458, 0.500])
V_H = np.array([-0.470, -0.600, -0.730, -0.860, -0.960, -1.100, -1.200, -1.300, -1.400, -1.500])

# Uncertainties
current_err = 0.005  # mA
V_H_err = 0.05       # mV

# Perform linear regression
# Using scipy.stats.linregress which provides slope, intercept, and standard errors
regress_result = stats.linregress(current, V_H)
slope = regress_result.slope
intercept = regress_result.intercept
slope_err = regress_result.stderr
# Note: linregress in recent SciPy versions returns intercept_stderr; if unavailable, one can calculate it separately.
intercept_err = regress_result.intercept_stderr if hasattr(regress_result, 'intercept_stderr') else None

# Create fitted line for plotting
current_fit = np.linspace(current.min(), current.max(), 100)
V_H_fit = slope * current_fit + intercept

# Plotting the data with error bars
plt.errorbar(current, V_H, xerr=current_err, yerr=V_H_err, fmt='o', label='Data with error bars', capsize=3, markersize=5)

# Plot the linear trend line
plt.plot(current_fit, V_H_fit, 'r-', label=f'Fit: V_H = ({slope:.3f}±{slope_err:.3f}) I + ({intercept:.3f}±{intercept_err:.3f})')

# Labeling the axes and title
plt.xlabel('Current (mA)')
plt.ylabel('Hall Voltage (mV)')
plt.title('Hall Voltage vs. Current with Linear Fit')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
