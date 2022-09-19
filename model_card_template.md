# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Nguyen Thai Vinh Phuc created this model. He trained random forest models on Census Bureau dataset.
Hyperparameters are defined in hyper_parameter.yaml.

## Intended Use
I solved a classification problem to predict if income is greater or less than $50k/yr in census data

## Training Data
UCI Census Dataset is cleaned by removig unneccessary spaces inside cvs file.
UCI Census Dataset is splitted intor training dataset and testing dataset.
Training data is 80% split of UCI Census Dataset.

## Evaluation Data
20% of UCI Census Dataset was used for evaluation .
The remaining of the data was used for training the model with cross validation

## Metrics
There are three metrics which are used for the evaluation for models such as precision, recall and fbeta.
Precision: 0.78
Recall: 0.62
Fbeta: 0.69

## Ethical Considerations
One of considerations is that the distribution of data features such as country, race and gender is 
not uniform. Therefore, it can make the models bias.


## Caveats and Recommendations
Next, we can address the unbalance dataset by upsampling the minor class. 
Furthermore, we can use PCA for Hyper-parameter tuning and feature selection.