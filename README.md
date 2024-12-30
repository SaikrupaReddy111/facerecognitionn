# facerecognitionn
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

def load_images(data_path):
    images = []
    labels = []
    for person_id in os.listdir(data_path):
        person_path = os.path.join(data_path, person_id)
        if os.path.isdir(person_path):
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (100, 100))  # Resize for consistency
                images.append(img.flatten())
                labels.append(int(person_id))
    return np.array(images), np.array(labels)

def eigenfaces(face_db, k):
    # Calculate mean face
    mean_face = np.mean(face_db, axis=0)
    
    # Mean zero faces
    zero_faces = face_db - mean_face
    
    # Calculate surrogate covariance matrix
    covariance = np.dot(zero_faces, zero_faces.T)
    
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    
    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top k eigenvectors
    feature_vectors = eigenvectors[:, :k]
    
    # Generate eigenfaces
    eigenfaces = np.dot(feature_vectors.T, zero_faces)
    
    # Generate signatures
    signatures = np.dot(eigenfaces, zero_faces.T)
    
    return eigenfaces, signatures, mean_face

def main():
    # Load dataset
    data_path = "dataset"  # Update with your dataset path
    images, labels = load_images(data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.4, random_state=42)
    
    accuracies = []
    k_values = range(5, 51, 5)
    
    for k in k_values:
        # Generate eigenfaces and signatures
        eigen_faces, signatures, mean_face = eigenfaces(X_train, k)
        
        # Train neural network
        clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
        clf.fit(signatures.T, y_train)
        
        # Test faces
        test_signatures = []
        for test_face in X_test:
            zero_face = test_face - mean_face
            test_sig = np.dot(eigen_faces, zero_face)
            test_signatures.append(test_sig)
            
        test_signatures = np.array(test_signatures)
        
        # Calculate accuracy
        accuracy = clf.score(test_signatures, y_test)
        accuracies.append(accuracy)
        
        print(f"Accuracy for k={k}: {accuracy}")
    
    # Plot accuracy vs k value
    plt.plot(k_values, accuracies)
    plt.xlabel('k value')
    plt.ylabel('Accuracy')
    plt.title('Face Recognition Accuracy vs Number of Eigenfaces')
    plt.show()

if __name__ == "__main__":
    main()
