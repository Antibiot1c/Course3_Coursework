def normalize_data(train_images, test_images):
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return train_images, test_images
