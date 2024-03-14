import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras import layers, models, config

from keras.preprocessing.image import load_img, img_to_array

save_file = 'unsere4_siamese_model.keras'
train_data_directory = 'small_png'
test_data_directory = 'test_png'
image_size = (128, 128)
num_epochs = 3
batch_size = 32

# Define your Siamese network creation function
def create_siamese_network(input_shape):
    model_input = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation='relu')(model_input)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    twin = models.Model(inputs=model_input, outputs=x)

    input_1 = layers.Input(shape=input_shape)
    input_2 = layers.Input(shape=input_shape)
    output_1 = twin(input_1)
    output_2 = twin(input_2)

    # Calculate L1 distance using Subtract layer and Abs layer
    subtracted = layers.Subtract()([output_1, output_2])
    l1_distance = layers.Activation('relu')(subtracted)

    # Add a Dense layer to get the final output
    siamese_output = layers.Dense(1, activation='sigmoid')(l1_distance)

    siamese_model = models.Model(inputs=[input_1, input_2], outputs=siamese_output)

    return twin, siamese_model


# Define your function to create pairs from directories
def create_pairs_from_directory(directory):
    pairs = []
    pair_labels = []

    classes = os.listdir(directory)

    for class_name in classes:
        class_path = os.path.join(directory, class_name)

        if os.path.isdir(class_path):  # Ensure it's a directory
            class_images = os.listdir(class_path)

            for i in range(len(class_images)):
                for j in range(i + 1, len(class_images)):
                    # Create pairs of images from the same class
                    pairs.append([os.path.join(class_path, class_images[i]), os.path.join(class_path, class_images[j])])
                    pair_labels.append(1)  # Similar pair

                    # Create pairs of images from different classes
                    other_class_name = np.random.choice([name for name in classes if name != class_name])
                    other_class_path = os.path.join(directory, other_class_name)
                    other_image = os.path.join(other_class_path, np.random.choice(os.listdir(other_class_path)))

                    pairs.append([os.path.join(class_path, class_images[i]), other_image])
                    pair_labels.append(0)  # Dissimilar pair

    return pairs, pair_labels

# Define your function to load and preprocess data
def load_and_preprocess_data(train_data_dir, test_data_dir, batch_size=5):
    train_pairs, train_labels = create_pairs_from_directory(train_data_dir)
    test_pairs, test_labels = create_pairs_from_directory(test_data_dir)

    def load_images(pair_paths, target_size = image_size):
        images = []
        img_width, img_height = None, None  # Initialize variables

        for pair in pair_paths:
            
            try:
                # Use the size of the first image in the pair to determine the size
                first_image = load_img(pair[0], target_size = target_size)  # Access the first path of the first element in the pair
            
            except FileNotFoundError as e:
                print(f"Error loading image: {e}. Check if the file exists for pair: {pair}")
                continue

            image_pair = [
                img_to_array(load_img(image_path, target_size=target_size)) / 255.0
                for image_path in pair
            ]

            # Ensure that both images in the pair have the same shape
            if image_pair[0].shape == image_pair[1].shape:
                images.append(image_pair)
            else:
                print(f"Ignoring pair {pair} due to shape mismatch. Shapes: {image_pair[0].shape}, {image_pair[1].shape}")

        return images, target_size[0], target_size[1]

    train_pairs, img_width, img_height = load_images(train_pairs)
    test_pairs, _, _ = load_images(test_pairs)

    return train_pairs, train_labels, test_pairs, test_labels, img_width, img_height

def load_and_preprocess_pair(image1_path, image2_path, target_size=image_size):
        img1 = img_to_array(load_img(image1_path, target_size=target_size)) / 255.0
        img2 = img_to_array(load_img(image2_path, target_size=target_size)) / 255.0

        return [img1, img2]

def build_network(): 
    if os.path.exists(save_file):
        # Load the pre-trained model if it exists
        siamese_model = tf.keras.models.load_model(save_file)
        print(f"Loaded pre-trained Siamese model from {save_file}")
    else:
        # Train the Siamese model
        # ... (your training code)
        # Load and preprocess data
        train_pairs, train_labels, test_pairs, test_labels, img_width, img_height = load_and_preprocess_data(train_data_directory, test_data_directory)

        # Build and compile Siamese network
        twin, siamese_model = create_siamese_network((image_size[0], image_size[1], 3))
        siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the Siamese model
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            for i in range(0, len(train_pairs), batch_size):
                pair_batch = train_pairs[i:i + batch_size]
                label_batch = np.array(train_labels[i:i + batch_size])

                # Convert to numpy array if not already
                pair_batch = np.array(pair_batch)

                # Ensure label_batch has the shape (batch_size, 1)
                label_batch = label_batch.reshape(-1, 1)

                loss = siamese_model.train_on_batch([pair_batch[:, 0], pair_batch[:, 1]], label_batch)

                print(f"Batch {i // batch_size + 1}/{len(train_pairs) // batch_size} - Loss: {loss}")


        # Evaluate the model on the test set
        total_accuracy = 0.0
        num_batches = len(test_pairs)

        for i in range(0, len(test_pairs), batch_size):
            pair_batch = test_pairs[i:i + batch_size]
            label_batch = test_labels[i:i + batch_size]

            # Convert to numpy array if not already
            pair_batch = np.array(pair_batch)

            # Extract the images from the pair batch
            images_1 = pair_batch[:, 0]
            images_2 = pair_batch[:, 1]

            # Predict the similarity scores
            similarity_scores = siamese_model.predict([images_1, images_2])

            # Assuming similarity_scores is a 2D array, extract the predicted labels
            predicted_labels = (similarity_scores > 0.5).astype(int)

            # Calculate accuracy
            accuracy = np.mean(predicted_labels == label_batch)
            total_accuracy += accuracy

        average_accuracy = total_accuracy / (len(test_pairs) // batch_size)
        print(f'Test Accuracy: {average_accuracy * 100:.2f}%')

        # Save the trained model
        siamese_model.save(save_file)
        print(f"Siamese model saved to {save_file}")

    return siamese_model
    
    
def main():
    siamese_model = build_network()
    
    # Example usage
    image1_path = 'izometrikler_vektörize/isoai-04.jpg'
    image2_path = 'png/armchair/491.png'

    image_pair = load_and_preprocess_pair(image1_path, image2_path)

    # Reshape the image pair to match the model's input shape
    image_pair_reshaped = [
        np.expand_dims(image_pair[0], axis=0),  # Add batch dimension
        np.expand_dims(image_pair[1], axis=0)   # Add batch dimension
    ]

    # Make a prediction using the loaded model
    similarity_raw_score = siamese_model.predict_on_batch(image_pair_reshaped)

    # Apply sigmoid activation to get similarity percentage
    similarity_percentage = 1 / (1 + np.exp(-similarity_raw_score))

    print(f"Similarity Percentage: {similarity_percentage[0][0] * 100:.2f}%")

if __name__ == "__main__":
    
    main()
    

#TODO make full data 
#image size
#[array(0.5821585, dtype=float32), array(0.77807343, dtype=float32)]