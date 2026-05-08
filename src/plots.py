import matplotlib.pyplot as plt

def plot_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].plot(history.history['loss'], label='Train Loss', color='blue', lw=2)
    axs[0].plot(history.history['val_loss'], label='Val Loss', color='red', lw=2)
    axs[0].set_title('Model Loss History')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss (MSE)')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)


    axs[1].plot(history.history['mae'], label='Train MAE', color='blue', lw=2)
    axs[1].plot(history.history['val_mae'], label='Val MAE', color='red', lw=2)
    axs[1].set_title('Model MAE History')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('MAE (Calories)')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.show()

plot_history(history)
