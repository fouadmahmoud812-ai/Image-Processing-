def plot_actual_vs_predicted(y_true, y_pred):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, color='orange')

    max_val = max(max(y_true), max(y_pred))
    plt.plot([0, max_val], [0, max_val], color='blue', linestyle='--', lw=2)

    plt.title('Actual vs Predicted Calories (Fusion Model)')
    plt.xlabel('Actual Calories')
    plt.ylabel('Predicted Calories')
    plt.grid(True, alpha=0.2)
    plt.show()

plot_actual_vs_predicted(y_true, y_pred)
