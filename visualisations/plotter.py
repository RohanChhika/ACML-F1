import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class Plotter:
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_loss(self, losses):
        epochs = list(range(1, len(losses) + 1))
        
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(epochs, losses, marker='o', linestyle='-', color='#FF6F61', linewidth=2, markersize=5, label='Training Loss')

        # Glow effect
        for alpha in [0.2, 0.1, 0.05]:
            ax.plot(epochs, losses, color='#FF6F61', linewidth=8, alpha=alpha)

        # Beautify the plot
        ax.set_title("Training Loss over Epochs", fontsize=18, weight='bold', color='#333333', pad=20)
        ax.set_xlabel("Epoch", fontsize=14, labelpad=10)
        ax.set_ylabel("MSE Loss", fontsize=14, labelpad=10)
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_xticks(range(0, len(epochs) + 1, 5))
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(fontsize=12)

        fig.patch.set_facecolor('#f9f9f9')
        ax.set_facecolor('#ffffff')
        plt.tight_layout()
        plt.show()
