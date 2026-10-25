# Emotion Recognition Using Machine Learning

## 1. Introduction

In this project, we trained machine learning models to recognize emotions when trained on face images. We worked with two datasets: FER2013 and KDEF. We tried different ways to extract the features and classify the emotions.

The comparison was between three methods: HOG, facial landmarks and PCA and we used three different classifiers: Logistic Regression, SVM and Random Forest. We also tested whether models trained on one dataset could work on the other.

## 2. Datasets

### 2.1 FER2013 [R1]

FER2013 contains 35,887 grayscale images and  each image has  a resolution of 48 × 48 pixels. The dataset contains 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise and Neutral.

The dataset was already divided into training, validation and testing:

| Split | Images |
|---|---:|
| Training | 28,709 |
| Validation | 3,589 |
| Test | 3,589 |

For our work, we used a smaller sample of 999 training images and 301 validation images in order to minimise the execution time. We used the full test set for evaluation.

During  the data exploration, we noticed that the emotions were not equally represented. Happy had the most images in our training sample with 251images, while Disgust had only 15. This is something that may affect how well the models learn to recognize less common emotions.

### 2.2 KDEF [R2]

KDEF contains images of people showing seven different facial expressions. Unlike FER2013, the images  include different camera angles.

Initially we had 4,900 KDEF files. Two files did not follow the standard emotion-image naming format and for that reason we  did not include them. This left us with 4,898 images from 140 subjects.

We divided the dataset into training, validation and testing sets based on subject IDs. We made sure that pictures of the same person did not appear in different sets. Below is the samary table of each dataset:

| Split | Images before preprocessing |
|---|---:|
| Training | 3,113 |
| Validation | 805 |
| Test | 980 |

For preprocessing we used OpenCV's Haar Cascade detector to detect and crop faces.Then we converted the images to grayscale, resized them to 48 × 48 pixels and prepared them for feature extraction.

From the results we came to the conclusion that the detector couldn't recognise emotions in side-profile pictures. To be more precise in the test set, 403 out of 980 images could not be processed, (mostly full left and full right views).

As a result, our KDEF test evaluation was based on 577 successfully processed images.

## 3. Methods

### 3.1 Feature Extraction

The feature extraction methods we explored were the following:

**HOG(Histogram of Oriented Gradients):** because it captures the edges and shapes of facial features, such as the eyes, eyebrows and mouth. It can help the model recognize changes in facial expressions, such as the shape of the mouth or eyebrows.

**Facial Landmarks:** We extracted 68 facial points (including points around the eyes, eyebrows, nose and mouth). Each point has an x and y coordinate which gave us 136 features per image.

**PCA:** For the experiments we used 100 principal components.

### 3.2 Classification Models

For each feature type, we trained three classifiers: Logistic Regression, SVM and Random Forest

This produced nine baseline combinations(HOG with Regression - HOG with SVM - HOG with Random Forest, Facial Landmarks with Regression etc.) for each dataset.

We used accuracy, balanced accuracy and macro F1 score to compare the models and focused mainly on macro F1 because it considers both precision and recall and gives equal importance to each emotion class, even when some classes have fewer images

## 4. FER2013 Experiments

### 4.1 Baseline Results

We trained all nine combinations using HOG, landmarks and PCA features.

Landmarks with Logistic Regression achieved the highest validation macro F1 score (0.290), while HOG with SVM achieved the highest accuracy (39.5%). The table bellow povides details regarding the metrcis of each model.

Some of the main results are shown below.

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Landmarks + LR | 38.2% | 0.290 |
| HOG + SVM | 39.5% | 0.288 |
| Landmarks + RF | 38.2% | 0.281 |
| Landmarks + SVM | 37.9% | 0.260 |
| PCA + RF | — | 0.180 |

From the results we can see that, HOG and landmark features performed better than PCA.

The scores are still relatively low. This shows that  showing recognizing facial emotions is not easy, especially with a small training sample.

### 4.2 Creative Challenge

For the creative challenge, we tried to improve Landmarks with Logistic Regression. For that reason we did the following changes:

1. Added balanced class weights.
2. Changed the regularization parameter.

We first tested each change separately and then by combining them.

| Variation | Accuracy | Macro F1 |
|---|---:|---:|
| Baseline | 38.2% | 0.290 |
| Class weights | 36.5% | 0.285 |
| Parameter tuning | **42.2%** | **0.314** |
| Both changes | 38.5% | 0.310 |

- Best results came from changing the regularization method which increased macro F1 to 0.314.

- Adding class weights alone did not improve the model. 

- Combining both changes also gave a slightly lower macro F1 than parameter tuning alone.

As a result we selected the tuned Landmarks + Logistic Regression model.

### 4.3 Final Test Results

We evaluated the selected model on the FER2013 test set containing 3,589 images.

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 42.19% | 38.98% |
| Balanced accuracy | 33.27% | 31.51% |
| Macro F1 | 0.314 | 0.308 |

The model performed best on Happy with F1 = 0.61 and Surprise where F1 = 0.44. It struggled most with Disgust with F1 = 0.03 and Fear with F1 = 0.18.

### 4.4 Error Analysis

For the error analysis we looked at misclassified images to understand the model's mistakes.

The most common errors were:

| Actual emotion | Predicted emotion | Images |
|---|---|---:|
| Sad | Neutral | 146 |
| Sad | Happy | 138 |
| Fear | Happy | 122 |
| Neutral | Sad | 122 |
| Angry | Happy | 111 |

These results show that facial landmarks alone may not capture enough information to identify specific expressions.

For example, slightly changes in the face expressions might be difficult for the model to recognize using only the positions of facial points.

The small training sample and the imbalance of the face expressions in the class may also have affected the results.

## 5. KDEF Experiments

### 5.1 Baseline Results

We followed the same approach using KDEF, testing all nine combinations of features and classifiers.

The best model was HOG with SVM, which achieved 61.0% accuracy and 0.599 macro F1.

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

We evaluated HOG + SVM on the held - out KDEF test data.

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 61.0% | 61.18% |
| Balanced accuracy | 60.4% | 60.75% |
| Macro F1 | 0.599 | 0.603 |

The results were very close between validation and test.

The model performed best on Happy with F1 = 0.85 and Disgust with F1 = 0.74. Fear with F1 = 0.36 and Sad with F1 = 0.42 were the most difficult.

However, these test results only include 577 images because again many side-profile images could not be processed by the face detector.

### 5.3 Cross-Dataset Evaluation

We also tested whether models trained on one dataset could recognize emotions in the other dataset.

We evaluated three feature–classifier combinations in both directions.

| Model | KDEF to FER2013 F1 | FER2013 to KDEF F1 |
|---|---:|---:|
| HOG + SVM | 0.169 | 0.153 |
| Landmarks + LR | **0.219** | **0.180** |
| PCA + SVM | 0.053 | 0.094 |

The results were much lower than when the models were evaluated on their original datasets.

Landmarks with Logistic Regression performed best in both cases, although the scores were still low.

One possible reason is the difference between the datasets. KDEF contains posed expressions in controlled conditions, while FER2013 contains more varied images.

Another reason why models struggle to generalize could the differences in image cropping, lighting and facial expressions.

These experiments show that a model performing well on one dataset does not necessarily perform well on another.

### 5.4 Creative Challenge

We tried to improve KDEF's best baseline, HOG + SVM, by changing two settings:

1. Added balanced class weights.
2. Increased the SVM parameter C from 1.0 to 3.0.

We tested both changes separately and together.

| Variation | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Baseline | 61.0% | 60.4% | 0.599 |
| Class weights | 62.1% | 62.2% | 0.612 |
| Tune C | **65.4%** | **65.3%** | **0.645** |
| Both changes | **65.4%** | **65.3%** | **0.645** |

- Increasing C gave the biggest improvement, raising macro F1 to 0.645.

- Adding class weights also improved the results slightly

- Combining both changes though, did not improve the scores further.

We selected HOG + SVM with C = 3.0 as our best creative-challenge model.

### 5.5 Error Analysis

For the error analysis here we examined some images that the tuned model classified incorrectly.

The most common mistakes were:

| Actual emotion | Predicted emotion | Images |
|---|---|---:|
| Sad | Angry | 17 |
| Disgust | Angry | 10 |
| Fear | Sad | 10 |
| Fear | Surprise | 9 |
| Angry | Disgust | 8 |

These mistakes suggest that some emotions have similar visual features.

For example, Fear and Surprise can both involve raised eyebrows and an open mouth, which might make them harder to identify.

Although tuning improved the overall performance, the model still made mistakes between several emotion classes.

## 6. Fuzzy classifier, Real-time demo

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

- All of these features were made from landmarks as well as support intermediate step such as eyes locations and their width. 

- All of the features are divided by the distance between the eyes as to make the face size and distance from the camera irrelevant. 

- The model is a Takagi-Sugeno model using pyFUME library:

1. It first uses fuzzy c-means clustering
2. then Gaussian membership functions 
3. and lastly a linear output per rule. 

The structure of the models is one-vs-rest. That means that all 7 emotion have their own model, each with 3 rules, and the emotion with the highest output value wins. 

The results on the given split are:

| Metric | Value | 
|---|---|
| accuracy | 0.68 |
| balanced_accuracy | 0.68 |
| macro f1 | 0.65 |

Overall performance of the human-understandble rules seems to be mixed. On one side, Happy emotion has a visibly highest smile_ratio, and the widest mouth, which matches the intuition. 

This is also the case for Surprise emotion (which has the highest value for open_mouth and brow_height), Disgust (smallest lip_raise) and Angry (lowest eyebrows). 

On the other hand, the Fear is often predicted as a surprise because both have an open mouth and eyes with raised eyebrows leading to misclassification. 

A similar situation is with Sad and Neutral which get mixed up because of closed mouth. Therefore these 6 human-understandble features perform well on some emotions and fail on others. 

This is consistent with our expectations as more features would be needed to properly distinguish each emotion.

### 6.2 Real time demo

The demo.py is an almost identical copy of provided demo code, with one exception - the model used. It uses final_selected_model.pkl which uses landmarks + tuned LR.

Overall changing the model did work. Happy and Surprise emotions performed the best. The other emotions had much worse accuracy - mostly predicted as neutral (sad, fear, disgust). The biggest problem was the flickering - the emotions were changing very quickly therefore it was hard to follow which emotions are predicted. 

## 7. Discussion and Limitations

Our experiments showed that the best feature extraction method depends on the dataset.

For FER2013, facial landmarks with Logistic Regression gave the highest baseline macro F1. 

For KDEF, HOG with SVM performed best.

The fuzzy classifier showed that only six simple geometric features can reach almost 68% accuracy on the KDEF dataset. Some emotions were recognized better than others.

We also found that parameter tuning improved both selected models. However, there were several limitations.

First, as mentioned we only used a small sample of the FER2013 training data, which may have limited model performance.

Second, KDEF preprocessing was not successful for all camera angles. Many full side-profile images were excluded because the face detector could not detect them.

Finally, the cross-dataset experiments showed a large drop in performance. This suggests that our models were sensitive to differences between the two datasets.


## 8. Conclusion

In this project, we compared different machine-learning methods for facial emotion recognition using FER2013 and KDEF datasets.

For FER2013, the best validation results came from facial landmarks with tuned Logistic Regression, reaching a macro F1 of 0.314.

For KDEF, HOG with tuned SVM performed best on validation data, reaching a macro F1 of 0.645.

The cross-dataset experiments showed that both datasets were quite different and that our models struggled to recognize emotions outside the dataset they were trained on.

A fuzzy classifier with six geometric, human-understandable features reached a macro F1 of 0.65 on KDEF while staying interpretable and feasible.

The real time demo works with the new model, although predicts well only Happy and Suprise emotions.

Overall, we found that feature extraction and model settings affect performance, but recognizing facial emotions reliably remains challenging.

For future work, we could:
1.  use more training images, 
2. improve face detection for side views, 
3. and test data augmentation or more advanced models to improve generalization.

## 9. AI Statement
During the preparation of this work, we used ChatGPT (OpenAI) to review the work, clarify assignment requirements that were not clear, help organize the workflow, and assist with debugging errors during implementation as well reviewing the code when needed. The following parts of the assignment were affected by AI tool usage: Phase_A_FER2013_Experiments. As a consequence, I take full responsibility for the content of my work.Also CLAUDE was used in order to help with the landmark to feature conversion for fuzzy classifier. The following parts of the assignment were affected
by AI tool usage: fuzzy_classifier.ipynb. After using both
tools/services, Mateusz Wojcik s2528916, Efsathia Lampraki s2409445 and Amelia Skolimowska s20262129 evaluated the validity of the tool’s outputs, including the
sources that generative AI tools have used, and edited the content as needed. As a
consequence, Mateusz Wojcik s2528916 take(s) full responsibility for the content of their work.

## 10. References

[R1] Goodfellow, I. J., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., Hamner, B., ... & Bengio,
Y. (2013, November). Challenges in representation learning: A report on three machine
learning contests. In International conference on neural information processing (pp. 117-124).
Berlin, Heidelberg: Springer Berlin Heidelberg.

[R2] Goeleven, E., De Raedt, R., Leyman, L., & Verschuere, B. (2008). The Karolinska directed
emotional faces: a validation study. Cognition and emotion, 22(6), 1094-1118.


## Appendix – Confusion matrices

### A.1 FER2013 baselines
<img src="results/appendix/Phase_A_FER2013_Experiments_cell17_1.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell17_2.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell17_3.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell17_4.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell17_5.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell17_6.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell17_7.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell17_8.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell17_9.png" width="32%">

### A.2 FER2013 creative challenge, Landmarks + LR
<img src="results/appendix/Phase_A_FER2013_Experiments_cell20_2.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell20_3.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell20_4.png" width="32%">

### A.3 FER2013 final model, test set
<img src="results/appendix/Phase_A_FER2013_Experiments_cell23_1.png" width="32%">

### A.4 KDEF baselines
<img src="results/appendix/Phase_A_FER2013_Experiments_cell49_1.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell53_1.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell55_1.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell61_1.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell62_1.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell63_1.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell69_1.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell71_1.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell73_1.png" width="32%">

### A.5 KDEF tuning, HOG + SVM
<img src="results/appendix/Phase_A_FER2013_Experiments_cell106_2.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell106_3.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell106_4.png" width="32%">

### A.6 KDEF final model (HOG + SVM), test set
<img src="results/appendix/Phase_A_FER2013_Experiments_cell84_1.png" width="32%">

### A.7 Cross-dataset
<img src="results/appendix/Phase_A_FER2013_Experiments_cell90_1.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell94_1.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell98_1.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell99_1.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell100_1.png" width="32%">
<img src="results/appendix/Phase_A_FER2013_Experiments_cell101_1.png" width="32%">

### A.8 Fuzzy classifier, KDEF test set
<img src="results/appendix/fuzzy_classifier_cell14_1.png" width="32%">

