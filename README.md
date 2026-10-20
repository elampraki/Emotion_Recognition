# Emotion Recognition — Assignment 2

This project explores facial emotion recognition using the FER2013 dataset. The aim is to compare different image features and machine-learning classifiers, then see whether simple changes can improve the results.

## Person A — FER2013 Experiments

### Dataset

FER2013 contains 35,887 grayscale face images of size 48 × 48 pixels. The images are labelled with seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

We used the dataset's original splits:

| Split | Images |
| --- | ---: |
| Training | 28,709 |
| Validation (PublicTest) | 3,589 |
| Test (PrivateTest) | 3,589 |

For the experiments reported here, I used a smaller sample of **999 training images** and **301 validation images** to reduce execution time. The final test used all 3,589 test images.

The emotions are not equally represented in the sampled training data. For example, Happy had 251 images, while Disgust had only 15. This may affect how well the classifiers learn the different emotions.

### Features and classifiers

I tested three types of features:

- **HOG** describes the direction of edges in an image.
- **Facial landmarks** use 68 face points (136 x/y values) to describe the position of facial features.
- **PCA** reduces image data to a smaller set of components.

Each feature type was tested with **Logistic Regression**, **Support Vector Machine (SVM)**, and **Random Forest**, giving nine baseline combinations.

I compared them using accuracy, balanced accuracy, and macro F1. Macro F1 is useful here because it gives each emotion equal weight, including the less common ones.

### Baseline results

The table below highlights some of the baseline results from the validation sample:

| Model | Accuracy | Balanced accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Landmarks + Logistic Regression | 38.21% | 30.33% | **0.290** |
| HOG + SVM | **39.53%** | 29.95% | 0.288 |
| Landmarks + Random Forest | 38.21% | 28.92% | 0.281 |
| Landmarks + SVM | 37.87% | 28.22% | 0.260 |
| PCA + Random Forest | — | — | 0.180 |

All nine baseline combinations were run. The complete results, including the remaining models, are in the experiments notebook / generated baseline results table.

**Landmarks + Logistic Regression** had the highest validation macro F1, although it was only slightly better than HOG + SVM. HOG + SVM had the highest accuracy. Overall, the models still found some emotions difficult to recognize.

### Creative challenge

I used Landmarks + Logistic Regression as the baseline and tried two changes: adding balanced class weights and changing the regularization parameter. I also tested both changes together.

| Change | Accuracy | Balanced accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Baseline | 38.21% | 30.33% | 0.290 |
| Class weights | 36.54% | 29.52% | 0.285 |
| **Tune parameter** | **42.19%** | 33.27% | **0.314** |
| Both changes | 38.54% | **33.65%** | 0.310 |

Parameter tuning gave the best validation macro F1 and accuracy. Class weights alone did not improve the results. I therefore selected **Landmarks + Logistic Regression with the tuned parameter**.

### Final test

The selected landmark model was evaluated on the 3,589 FER2013 test images.

| Metric | Validation | Test |
| --- | ---: | ---: |
| Accuracy | 42.19% | 38.98% |
| Balanced accuracy | 33.27% | 31.51% |
| Macro F1 | 0.314 | 0.308 |

Happy was recognized most successfully (test F1 = 0.61), followed by Surprise (F1 = 0.44). The model struggled with Fear (F1 = 0.18) and especially Disgust (F1 = 0.03).

### Error analysis

I looked at incorrectly classified images and counted the most common confusion pairs:

| Actual emotion | Predicted as | Images |
| --- | --- | ---: |
| Sad | Neutral | 146 |
| Sad | Happy | 138 |
| Fear | Happy | 122 |
| Neutral | Sad | 122 |
| Angry | Happy | 111 |

These mistakes suggest that some expressions are difficult to distinguish using landmark positions alone. Disgust does not appear in this top-five list because it has only 55 images in the test set; its recall was still very low (0.02).

### Limitations and conclusion

The model's performance is limited, especially for the less common emotions. Using only 999 training images probably reduced how much the models could learn. Facial landmarks also describe face shape and positions, but may miss other visual details that matter for emotion recognition.

The test set was also viewed during an earlier HOG + SVM experiment, before all nine baseline comparisons were finalized. The landmark model was selected using validation results, but the later test evaluation should not be treated as completely untouched independent confirmation.

Overall, **landmark features with tuned Logistic Regression** gave the best validation macro F1 in these experiments. The results show that a simple model can recognize some facial expressions, but performance across all seven emotions is still weak.

### Files and reproducibility

- `src/features.py`: shared feature extraction and evaluation functions.
- `faceAndLandmarkModels/`: pretrained facial landmark files.
- `z_load_split_fer2013.ipynb`: dataset preparation notebook.
- `baseline_fer2013_landmarks.ipynb`: provided baseline example.

The Person A experiments notebook, generated results, and final trained model should be added to the repository as part of the submission if they are not already committed. The FER2013 data files are stored locally under `data/fer/` and are excluded from Git.

To reproduce the reported runs, use the original FER2013 Training/PublicTest/PrivateTest splits and the same 999/301 sampled subsets (random seed 42). Results may vary if the sample, model parameters, or dependencies change.
