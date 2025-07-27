from tensorflow.keras.models import load_model

# === Load the Keras Model ===
model_path = "models/full_keras_model.keras"
model = load_model(model_path)

# === Print Input Shape ===
print("📥 Input Shape:", model.input_shape)

# === Print Output Shape ===
print("📤 Output Shape:", model.output_shape)

# === Optional: Layer Summary
print("\n🧠 Model Architecture:")
model.summary()