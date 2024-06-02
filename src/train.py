import tensorflow as tf
from data.dataset import load_data
from src.utils.data_preprocessing import normalize_data
from src.models.cnn_model import create_model

# Load and preprocess data
(train_images, train_labels), (test_images, test_labels) = load_data()
train_images, test_images = normalize_data(train_images, test_images)

# Create and train model
model = create_model()
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# Save model
model.save('cnn_model.h5')
