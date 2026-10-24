# Emotion Recognition Using Machine Learning

## 1. Introduction

In this project, we explored how machine learning can be used to recognize emotions from facial images. We worked with two datasets, FER2013 and KDEF, and compared different features and classification models.

We used three feature extraction methods: Histogram of Oriented Gradients (HOG), facial landmarks and Principal Component Analysis (PCA). For classification, we tested Logistic Regression, Support Vector Machine (SVM) and Random Forest.

Our goal was to find which combinations worked best and see how well the models could recognize emotions from a different dataset.

## 2. Datasets

### 2.1 FER2013 — Person A

FER2013 contains 35,887 grayscale images, each with a resolution of 48 × 48 pixels. The images are divided into seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise and Neutral.

The dataset was already divided into training, validation and test sets:

| Split | Images |
|---|---:|
| Training | 28,709 |
| Validation | 3,589 |
| Test | 3,589 |

For our experiments, we used a smaller sample of 999 training images and 301 validation images to reduce the execution time. We kept the full test set for evaluation.

During data exploration, we noticed that the emotions were not equally represented. Happy had the most images in our training sample (251), while Disgust had only 15. This imbalance could affect how well the models learn to recognize less common emotions.

### 2.2 KDEF — Person B

KDEF contains images of people showing seven different facial expressions. Unlike FER2013, the images were taken in controlled conditions and include different camera angles.

We started with 4,900 KDEF files. Two files did not follow the standard emotion-image naming format, so we excluded them. This left us with 4,898 images from 140 subjects.

We divided the dataset into training, validation and test sets based on subject IDs, making sure images of the same person did not appear in different sets.

| Split | Images before preprocessing |
|---|---:|
| Training | 3,113 |
| Validation | 805 |
| Test | 980 |

Before training, we used OpenCV's Haar Cascade detector to detect and crop faces. We then converted the images to grayscale, resized them to 48 × 48 pixels and prepared them for feature extraction.

However, the detector had difficulties with side-profile images. In the test set, 403 out of 980 images could not be processed, mainly full left and full right views.

As a result, our KDEF test evaluation was based on 577 successfully processed images.

## 3. Methods

### 3.1 Feature Extraction

We tested three feature extraction methods:

**HOG:** This method captures edges and their directions in an image. It can help the model recognize changes in facial expressions, such as the shape of the mouth or eyebrows.

**Facial Landmarks:** We used dlib to extract 68 facial points, including points around the eyes, eyebrows, nose and mouth. Each point has an x and y coordinate, giving us 136 features per image.

**PCA:** We used PCA to reduce the number of image features while keeping important information. For our experiments, we used 100 principal components.

### 3.2 Classification Models

For each feature type, we trained three classifiers:

- **Logistic Regression:** A relatively simple model used for classification.
- **SVM:** A model that tries to separate different classes based on their features.
- **Random Forest:** A model that combines predictions from several decision trees.

This gave us nine baseline combinations for each dataset.

We compared the models using accuracy, balanced accuracy and macro F1 score. We focused mainly on macro F1 because it gives equal importance to each emotion class, even when some classes have fewer images.

## 4. Person A — FER2013 Experiments

### 4.1 Baseline Results

We trained all nine combinations using HOG, landmarks and PCA features.

Landmarks with Logistic Regression achieved the highest validation macro F1 score (0.290), while HOG with SVM achieved the highest accuracy (39.5%).

Some of the main results are shown below.

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Landmarks + LR | 38.2% | 0.290 |
| HOG + SVM | 39.5% | 0.288 |
| Landmarks + RF | 38.2% | 0.281 |
| Landmarks + SVM | 37.9% | 0.260 |
| PCA + RF | — | 0.180 |

Overall, HOG and landmark features performed better than PCA in our experiments.

The scores were still relatively low, showing that recognizing facial emotions is not an easy classification task, especially with a small training sample.

### 4.2 Creative Challenge

For the creative challenge, we tried to improve Landmarks + Logistic Regression by making two changes:

1. Adding balanced class weights.
2. Changing the regularization parameter.

We tested each change separately and then combined them.

| Variation | Accuracy | Macro F1 |
|---|---:|---:|
| Baseline | 38.2% | 0.290 |
| Class weights | 36.5% | 0.285 |
| Parameter tuning | **42.2%** | **0.314** |
| Both changes | 38.5% | 0.310 |

Changing the regularization parameter gave the best results, increasing macro F1 from 0.290 to 0.314.

Adding class weights alone did not improve the model. Combining both changes also gave a slightly lower macro F1 than parameter tuning alone.

We therefore selected the tuned Landmarks + Logistic Regression model.

### 4.3 Final Test Results

We evaluated the selected model on the FER2013 test set containing 3,589 images.

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 42.19% | 38.98% |
| Balanced accuracy | 33.27% | 31.51% |
| Macro F1 | 0.314 | 0.308 |

The model performed best on Happy (F1 = 0.61) and Surprise (F1 = 0.44). It struggled most with Disgust (F1 = 0.03) and Fear (F1 = 0.18).

### 4.4 Error Analysis

We looked at misclassified images to understand the model's mistakes.

The most common errors were:

| Actual emotion | Predicted emotion | Images |
|---|---|---:|
| Sad | Neutral | 146 |
| Sad | Happy | 138 |
| Fear | Happy | 122 |
| Neutral | Sad | 122 |
| Angry | Happy | 111 |

These results suggest that facial landmarks alone may not capture enough information to distinguish between some expressions.

For example, subtle changes in the face might be difficult to recognize using only the positions of facial points.

The small training sample and the class imbalance may also have affected the results.

## 5. Person B — KDEF Experiments

### 5.1 Baseline Results

We followed the same approach using KDEF, testing all nine combinations of features and classifiers.

The best model was HOG with SVM, which achieved 61.0% validation accuracy and 0.599 macro F1.

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| HOG + SVM | **61.0%** | **0.599** |
| PCA + SVM | 58.1% | 0.578 |
| HOG + RF | 56.8% | 0.552 |
| PCA + LR | 50.0% | 0.487 |
| PCA + RF | 48.5% | 0.475 |
| HOG + LR | 47.8% | 0.470 |
| Landmarks + RF | 34.8% | 0.332 |
| Landmarks + LR | 31.3% | 0.294 |
| Landmarks + SVM | 32.2% | 0.293 |

HOG and PCA performed better than facial landmarks on KDEF. This may be because they capture more information about the appearance of facial expressions, while landmarks mainly focus on the positions of facial features.

### 5.2 KDEF Test Results

We evaluated HOG + SVM on the held-out KDEF test subjects.

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 61.0% | 61.18% |
| Balanced accuracy | 60.4% | 60.75% |
| Macro F1 | 0.599 | 0.603 |

The results were very close between validation and test.

The model performed best on Happy (F1 = 0.85) and Disgust (F1 = 0.74). Fear (F1 = 0.36) and Sad (F1 = 0.42) were the most difficult.

However, these test results only include 577 images because many side-profile images could not be processed by the face detector.

### 5.3 Cross-Dataset Evaluation

We also tested whether models trained on one dataset could recognize emotions in the other dataset.

We evaluated three feature–classifier combinations in both directions.

| Model | KDEF → FER2013 F1 | FER2013 → KDEF F1 |
|---|---:|---:|
| HOG + SVM | 0.169 | 0.153 |
| Landmarks + LR | **0.219** | **0.180** |
| PCA + SVM | 0.053 | 0.094 |

The results were much lower than when the models were evaluated on their original datasets.

Landmarks with Logistic Regression performed best in both transfer directions, although its scores were still low.

One possible reason is the difference between the datasets. KDEF contains posed expressions in controlled conditions, while FER2013 contains more varied images.

Differences in image cropping, lighting and facial expressions may also explain why the models struggled to generalize.

These experiments show that a model performing well on one dataset does not necessarily perform well on another.

### 5.4 Creative Challenge

We tried to improve KDEF's best baseline, HOG + SVM, by changing two settings:

1. Adding balanced class weights.
2. Increasing the SVM parameter C from 1.0 to 3.0.

We tested both changes separately and together.

| Variation | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Baseline | 61.0% | 60.4% | 0.599 |
| Class weights | 62.1% | 62.2% | 0.612 |
| Tune C | **65.4%** | **65.3%** | **0.645** |
| Both changes | **65.4%** | **65.3%** | **0.645** |

Increasing C gave the biggest improvement, raising macro F1 from 0.599 to 0.645.

Adding class weights also improved the results slightly, but combining both changes did not improve the scores further.

We selected HOG + SVM with C = 3.0 as our best creative-challenge model.

### 5.5 Error Analysis

We examined some images that the tuned model classified incorrectly.

The most common mistakes were:

| Actual emotion | Predicted emotion | Images |
|---|---|---:|
| Sad | Angry | 17 |
| Disgust | Angry | 10 |
| Fear | Sad | 10 |
| Fear | Surprise | 9 |
| Angry | Disgust | 8 |

These mistakes suggest that some emotions have similar visual features.

For example, Fear and Surprise can both involve raised eyebrows and an open mouth, which might make them harder to distinguish.

Although tuning improved the overall performance, the model still made mistakes between several emotion classes.

## 6. Person C - Fuzzy classifier, Real-time demo

### 6.1 Fuzzy classifier
The fuzzy classifier was trained on the KDEF dataset mentioned in the ealier sections. Using dlib landmarks we create 6 geometric features:

| Feature | What is measured | 
|---|---|
| mouth_open | gap between inner lip points |
| mouth_width | corner to corner of mouth |
| eye_open | gap between eyelids |
| brow_height | eyebrows above eyes |
| smile_ratio | corners above the mouth centre divided by mouth width|
|lip_raise |nose tip to upper lip|

All of these features were made from landmars as well as support intermediate step such as eyes locations and their width. All of the features are divided by the distance between the eyes as to make the faze size and distance from the camera irrelevant. The model is a Takagi-Sugeno model using pyFUME library. It first usess fuzzy c-means clustering, then Gaussian membership function and lastly a linear output per rule. The strucure of the models is one-vs-rest, meaning that all 7 emotion have their own model, each with 3 rules, and the emothion with the highest output value wins. The results on the given split are:

| Metric | Value | 
|---|---|
| accuracy | 0.68 |
| balanced_accuracy | 0.68 |
| macro f1 | 0.65 |

Overall performance of the human-understable rules seems to be mixed. On one side, Happy emotion has a visibly highest smile_ratio, and the widest mouth, which matches the intuition. This is also the case for Suprise emotion (which has the highest value for open_mouth and brow_height), Disgust (smallest lip_raise) and Angry(lowest eyebrows). On the other hand, the Fear is often predicted as a suprise because both have an open mouth and eyes with raised eyebrows leading to misclassification. A similar situation is with Sad and Neutral which get mixed up because of closed mouth. Therefore these 6 human-understable features perform well on some emotions and fail on others. This is consistent with our expectations as more features would be needed to properly distinguish each emotion.

### 6.2 Real time demo

The demo.py is a almost identical copy of provided demo code, with one exception - the model used. It uses final_selected_model.pkl which uses landmarks + tuned LR.

Overall changing the model did work. Happy and suprised emotions performed the best. Thee other emotions had much worse accuracy - mostly predicted as neutral (sad, fear, disgust). The biggest problem was the flickering - the emotions were changing very quickly therefore it was hard to follow which emotions are predicted. 

## 7. Discussion and Limitations

Our experiments showed that the best feature extraction method depends on the dataset.

For FER2013, facial landmarks with Logistic Regression gave the highest baseline macro F1. For KDEF, HOG with SVM performed best.

We also found that parameter tuning improved both selected models.

However, there were several limitations.

First, we only used a small sample of the FER2013 training data, which may have limited model performance.

Second, KDEF preprocessing was not successful for all camera angles. Many full side-profile images were excluded because the face detector could not detect them.

Finally, the cross-dataset experiments showed a large drop in performance. This suggests that our models were sensitive to differences between the two datasets.

We had also examined test results during earlier experiments before all model comparisons were finalized. Because of this, the later test evaluations should not be considered completely independent of the development process.

## 8. Conclusion

In this project, we compared different machine-learning methods for facial emotion recognition using FER2013 and KDEF.

For FER2013, the best validation results came from facial landmarks with tuned Logistic Regression, reaching a macro F1 of 0.314.

For KDEF, HOG with tuned SVM performed best on validation data, reaching a macro F1 of 0.645.

The cross-dataset experiments showed that both datasets were quite different and that our models struggled to recognize emotions outside the dataset they were trained on.

Overall, we found that feature extraction and model settings affect performance, but recognizing facial emotions reliably remains challenging.

For future work, we could use more training images, improve face detection for side views, and test data augmentation or more advanced models to improve generalization.