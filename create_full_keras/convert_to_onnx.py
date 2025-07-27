import tensorflow as tf
import tf2onnx
import numpy as np
import os

# === Load Keras Model ===
model = tf.keras.models.load_model("models/full_keras_model.keras")

# === Infer input shape ===
input_shape = model.input_shape[1]  # Feature count from TF-IDF

# === Build Input Spec ===
spec = (tf.TensorSpec((None, input_shape), tf.float32, name="input"),)
model_func = tf.function(lambda x: model(x))

# === Convert to ONNX ===
os.makedirs("models", exist_ok=True)
onnx_model, _ = tf2onnx.convert.from_function(
    model_func,
    input_signature=spec,
    opset=13,
    output_path="models/full_keras_model.onnx"
)

print("✅ Converted to ONNX: models/full_keras_model.onnx")