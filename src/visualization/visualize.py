import matplotlib.pyplot as plt


def plot_model_history(history):
    # Plot training & validation accuracy values
    final_val_acc = history.history['val_acc'][-1]
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(f'Model accuracy - final acc {final_val_acc:.3f}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
