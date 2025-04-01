import pandas as pd
import matplotlib.pyplot as plt

# Read CSV files (adjust the file paths if necessary)
df_visco = pd.read_csv('./Coursework3/visco_sweep.csv')
df_standard = pd.read_csv('./Coursework3/standard_sweep.csv')

# Plot test error percentage vs. number of hidden units for both models
plt.figure(figsize=(6, 4))
plt.plot(df_visco['n_hidden'], df_visco['test_error_percentage'], label='Viscoelastic RNO', marker='o')
plt.plot(df_standard['n_hidden'], df_standard['test_error_percentage'], label='Standard RNO', marker='s')
plt.xlabel('Number of Internal Variables')
plt.ylabel('Test Error (%)')
# plt.title('Test Error Percentage vs. Internal Variables')
plt.legend()
plt.grid(True)

# Save the figure to the current folder
plt.savefig('./Coursework3/test_error_sweep.png', dpi=300, bbox_inches='tight')
plt.close()