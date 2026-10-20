from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix,ConfusionMatrixDisplay
from pathlib import Path
from skimage.feature import hog

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


#FER has a numerical labels
KDEF_TO_FER = {'angry': 0, 'disgust': 1, 'afraid': 2, 'happy': 3,
               'sad': 4, 'surprise': 5, 'neutral': 6}

def fer_images(df):
    """Turn the 'pixels' column of a FER dataframe into an array of 48x48 images (0-255)."""
    return np.array([np.array(p.split(), dtype=np.uint8).reshape(48, 48) for p in df['pixels']])

def hog_features(images):
    """Extract HOG features from an array of grayscale images."""
    return np.asarray([hog(img.astype(np.float32) / 255.0,
                           orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), block_norm='L2-Hys')
                       for img in images], dtype=np.float32)

def landmark_features(images, predictor_path='faceAndLandmarkModels/shape_predictor_68_face_landmarks_GTX.dat'):
    """Extract 68 face landmarks from each 48x48 image; returns 136 numbers/image."""
    import dlib
    # Find the model from the project folder
    model_path = (
        Path(__file__).resolve().parent.parent
        / "faceAndLandmarkModels"
        / "shape_predictor_68_face_landmarks_GTX.dat"
    )

    predictor = dlib.shape_predictor(str(model_path))
    rectangle = dlib.rectangle(0, 0, 47, 47)
    features = []
    for img in images:
        if img.max() <= 1:  # KDEF files can store pixels between 0 and 1
            img = img * 255
        # Make sure the image has the format dlib expects
        img = np.ascontiguousarray(img, dtype=np.uint8)
        shape = predictor(img, rectangle)
        xy = np.array([(p.x, p.y) for p in shape.parts()], dtype=np.float32)
        xy -= xy.mean(axis=0)
        scale = np.max(np.abs(xy), axis=0)
        xy /= np.where(scale == 0, 1, scale)
        features.append(xy.ravel())
    return np.asarray(features, dtype=np.float32)

def evaluate(y_true, y_pred, title):
    """Print the 3 metrics from the instrucitons (accuracy, blan)and show the confusion matrix."""

    #Metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    #Plotting the metrics
    cm = confusion_matrix(y_true, y_pred, labels=range(7)) # to ensure the confusion matrix always has all seven emotion classes
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title(title); plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.show()

    #Return a dict 
    return {'model': title, 'acc': accuracy, 'bal_acc': balanced_accuracy, 'macro_f1': macro_f1}
