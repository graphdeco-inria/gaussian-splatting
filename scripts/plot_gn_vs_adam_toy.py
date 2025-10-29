import numpy as np
import matplotlib.pyplot as plt

gn_steps = np.arange(500, 525)
gn_losses = [14.615, 14.67, 14.698, 14.714, 14.737, 14.767, 14.806, 14.814, 14.834, 14.839, 14.855, 14.868, 14.882, 14.884, 14.89, 14.898, 14.89, 14.915, 14.925, 14.936, 14.96, 14.975, 14.99, 15.013, 15.023]

adam_steps = np.arange(500, 901, 100)
adam_losses = [14.615, 14.856, 14.967, 15.066, 15.181]

plt.figure(figsize=(10, 6))
plt.plot(gn_steps, gn_losses, marker='o', label='Gauss-Newton', color='blue')
plt.plot(adam_steps, adam_losses, marker='s', label='Adam', color='orange')
plt.xlabel('Training Step')
plt.ylabel('Training Loss')
plt.title('Training Loss: Gauss-Newton vs Adam')
plt.legend()
plt.grid(True)
plt.savefig('figures/gn_vs_adam_training_loss.png')
