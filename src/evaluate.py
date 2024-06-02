import tensorflow as tf
from data.dataset import load_data
from src.utils.data_preprocessing import normalize_data
from src.utils.visualization import plot_sample_images

# Load and preprocess data
(train_images, train_labels), (test_images, test_labels) = load_data()
train_images, test_images = normalize_data(train_images, test_images)

# Load model
model = tf.keras.models.load_model('cnn_model.h5')

# Evaluate model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc}")

# Predict and visualize
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

print(f"Predicted class: {class_names[np.argmax(predictions[0])]}")
print(f"True class: {class_names[test_labels[0][0]]}")
