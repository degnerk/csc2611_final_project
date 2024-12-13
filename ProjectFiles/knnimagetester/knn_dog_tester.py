#Accuracy for KNN Model
import os
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import argparse

def extract_image_features(image_path):
    """
    Extract features from a single image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        features: numpy array of image features
    """
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        # Extract color features
        color_means = img_array.mean(axis=(0,1))
        color_stds = img_array.std(axis=(0,1))
        hist_r = np.histogram(img_array[:,:,0], bins=32, range=(0,256))[0]
        hist_g = np.histogram(img_array[:,:,1], bins=32, range=(0,256))[0]
        hist_b = np.histogram(img_array[:,:,2], bins=32, range=(0,256))[0]
        
        # Combine features
        features = np.concatenate([
            color_means,
            color_stds,
            hist_r,
            hist_g,
            hist_b
        ])
        
        return features.reshape(1, -1)
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def train_and_save_model(data_path, model_save_path):
    """
    Train the KNN model and save it along with the scaler.
    """
    # Load and prepare training data
    print("Loading training data...")
    X_train, y_train = load_image_dataset(data_path, 'train')
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    print("Training model...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    print(f"Saving model to {model_save_path}")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump({
        'model': knn,
        'scaler': scaler,
        'classes': list(knn.classes_)
    }, model_save_path)

def predict_breed(image_path, model_path):
    """
    Predict the breed of a dog in a given image.
    
    Args:
        image_path: Path to the image file
        model_path: Path to the saved model file
        
    Returns:
        tuple: (predicted breed, confidence score)
    """
    # Load model and scaler
    saved_data = joblib.load(model_path)
    model = saved_data['model']
    scaler = saved_data['scaler']
    
    # Extract features from image
    features = extract_image_features(image_path)
    if features is None:
        return None, None
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Get predictions and distances
    distances, indices = model.kneighbors(features_scaled)
    
    # Convert distances to confidence scores (1 / (1 + distance))
    confidence_scores = 1 / (1 + distances[0])
    
    # Get the predicted class and confidence
    predicted_breed = model.predict(features_scaled)[0]
    confidence = confidence_scores.mean() * 100  # Convert to percentage
    
    return predicted_breed, confidence

def main():
    parser = argparse.ArgumentParser(description='Predict dog breed from image')
    parser.add_argument('--image', required=True, help='Path to the dog image')
    parser.add_argument('--model', default='models/dog_breed_knn.joblib', 
                      help='Path to the saved model file')
    parser.add_argument('--train', action='store_true',
                      help='Train a new model before prediction')
    parser.add_argument('--data', help='Path to training data directory')
    
    args = parser.parse_args()
    
    if args.train:
        if not args.data:
            print("Error: --data parameter is required for training")
            return
        train_and_save_model(args.data, args.model)
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return
        
    # Predict breed
    breed, confidence = predict_breed(args.image, args.model)
    
    if breed is not None:
        print(f"\nPredicted breed: {breed}")
        print(f"Confidence: {confidence:.2f}%")
    else:
        print("Failed to process image")

if __name__ == "__main__":
    main()