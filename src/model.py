import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

input_rgb = Input(shape=(224, 224, 3), name="rgb_input")
base_resnet = ResNet50(weights='imagenet', include_top=False)(input_rgb)

input_rgb = Input(shape=(224, 224, 3), name="rgb_input")
base_resnet = ResNet50(weights='imagenet', include_top=False)(input_rgb)
pool_rgb = GlobalAveragePooling2D()(base_resnet)

input_depth = Input(shape=(224, 224, 1), name="depth_input")

d = Conv2D(32, (3,3), activation='relu', padding='same')(input_depth)
d = MaxPooling2D()(d)
d = Conv2D(64, (3,3), activation='relu', padding='same')(d)
d = MaxPooling2D()(d)
d = GlobalAveragePooling2D()(d)

combined = Concatenate()([pool_rgb, d])
x = Dense(512, activation='relu')(combined)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
output = Dense(1)(x)

model_fusion = Model(inputs=[input_rgb, input_depth], outputs=output)
model_fusion.compile(optimizer=Adam(1e-4), loss='mae', metrics=['mae'])
