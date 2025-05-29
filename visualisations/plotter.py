import matplotlib.pyplot as plt

class Plotter:

    def plot_loss(self, losses):
        epochs = list(range(1, len(losses) + 1))

        plt.plot(epochs, losses, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss per Epoch")
        plt.legend()
        plt.show()

    