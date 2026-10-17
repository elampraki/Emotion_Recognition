import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


#FER has a numerical labels
KDEF_TO_FER = {'angry': 0, 'disgust': 1, 'afraid': 2, 'happy': 3,
               'sad': 4, 'surprise': 5, 'neutral': 6}

def fer_images(df):
    """Turn the 'pixels' column of a FER dataframe into an array of 48x48 images (0-255)."""
    return np.array([np.array(p.split(), dtype=np.uint8).reshape(48, 48) for p in df['pixels']])

def evaluate(y_true, y_pred, title):
    """Print the 3 metrics from the instrucitons (accuracy, blan)and show the confusion matrix."""

    #Metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    #Plotting the metrics
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title(title); plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.show()

    #Return a dict 
    return {'model': title, 'acc': accuracy, 'bal_acc': balanced_accuracy, 'macro_f1': macro_f1}