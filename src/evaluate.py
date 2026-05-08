
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

def get_clean_metrics(name, model, gen, steps):
    all_preds = []
    all_labels = []


    for i in range(steps):
        x_batch, y_batch = next(gen)
        preds = model.predict(x_batch, verbose=0)
        all_preds.extend(preds.flatten())
        all_labels.extend(y_batch)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)


    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)


    print(f"{name} Results:")
    print(f"  MAE : {mae:.2f} calories")
    print(f"  R2  : {r2:.4f}")
    print("-" * 20)

    return all_labels, all_preds


train_gen_eval = data_generator(train_df, batch_size=BATCH_SIZE)
get_clean_metrics("Train", model_fusion, train_gen_eval, steps_per_epoch)


val_gen_eval = data_generator(val_df, batch_size=BATCH_SIZE)
get_clean_metrics("Validation", model_fusion, val_gen_eval, validation_steps)


test_steps = len(test_df) // BATCH_SIZE
test_gen_eval = data_generator(test_df, batch_size=BATCH_SIZE)
y_true, y_pred = get_clean_metrics("Test", model_fusion, test_gen_eval, test_steps)


mask = y_true > 10
mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
print(f"Final Test MAPE: {mape:.2f}%")
