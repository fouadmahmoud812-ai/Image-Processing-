from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

print("Fusion Model Evaluation")


all_preds = []
all_labels = []


test_steps = len(test_df) // BATCH_SIZE
test_gen = data_generator(test_df, batch_size=BATCH_SIZE)

print("Predicting")
for i in range(test_steps):
    x_batch, y_batch = next(test_gen)
    preds = model_fusion.predict(x_batch, verbose=0)
    all_preds.extend(preds.flatten())
    all_labels.extend(y_batch)


all_preds = np.array(all_preds)
all_labels = np.array(all_labels)


mae = mean_absolute_error(all_labels, all_preds)
r2 = r2_score(all_labels, all_preds)


print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2 Score)     : {r2:.4f}")


