# Machine learning to predict clinical outcomes from RNS System background ECoG


## Data
Retrospective ECoG data from from patients implanted with the RNS System at NYU Langone Health. For the purpose of the project, we will be training and evaluating our model using only scheduled ECoG (background ECoF). The lasting time of each EcoG ranges from 120 seconds to 180 seconds. 


## Methods

### Feature Extraction
Band-power features were calculated using Fourier Transform for each ECoG segment for classic frequency bands falling within the hardware filter bandpass (4-90 Hz): Theta (4 - 8 Hz), Alpha (8 - 12 Hz), Beta (12 - 25 Hz), Low gamma (25 - 50 Hz), High gamma (50 - 90 Hz), and broadband activity (0. 01 - 90 Hz). 

### Machine Learning Models
Machine learning algorithms, including logistic regression, random forest, gradient boosting, SVM etc. were systematically trained on 80% of the entire dataset based on the above background ECoG features. Ten-fold cross-validation was applied to grid-search for optimal hyperparameters. The resulting classifiers were then tested on the remaining 20% of data. We further analyzed the effect of sleep on classification accuracy by separating ECoG segments recorded during sleep (e. g. 4am) from those recorded during awake hours (e. g. 10am, 4pm, 10pm from the same patient). An additional three classifiers were trained 1) with only sleep ECoG segment, 2) only awake segments, and 3) with all ECoGs segments but by adding an additional binary feature of sleep versus awake.


## Implement

The Matlab code for feature extraction is in the folder src_matlab.
The Python code for data preprocessing, feature extraction, model building, cross-validation, model evaluation,etc. is in the folder src.

Train the model and perform parameter tuning in the Jupyter notebook file ML_part.ipynb.
Evaluate the model and visualize the results in the Jupyter notebook ML results.ipynb



