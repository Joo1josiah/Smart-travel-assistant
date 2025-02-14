from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import save_model

# Load the MobileNetV2 pre-trained model
model = MobileNetV2(weights='imagenet')

# Save it to the models directory
model.save("backend/models/cnn_model.h5")

print("Model downloaded and saved successfully!")
