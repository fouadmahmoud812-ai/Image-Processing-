from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

BATCH_SIZE=16
train_gen = data_generator(train_df, batch_size=BATCH_SIZE)
val_gen = data_generator(val_df, batch_size=BATCH_SIZE)

steps_per_epoch = len(train_df) // BATCH_SIZE
validation_steps = len(val_df) // BATCH_SIZE

reduce_lr = ReduceLROnPlateau(
    monitor='val_mae',
    factor=0.5,
    patience=5,
    min_lr=1e-8,
    mode='min',
    verbose=1
)


checkpoint = ModelCheckpoint(
    'best_fusion_model.keras',
    monitor='val_mae',
    save_best_only=True,
    mode='min',
    verbose=1
)

callbacks_list = [reduce_lr, checkpoint]

print('Start Train ')

history = model_fusion.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=60,
    callbacks=callbacks_list,
    verbose=1
)
