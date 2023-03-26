# Heart Searcher

#### This project aims to establish the indicative conditions, features, and patterns of a sufficient production of human cardiomocytes from HPSCs. Differentiation causes cells to undergo major changes in their shape, size, and metabolism. These concepts served as the guide for which parameters in the dataset were most likely to be biologically relevant, with the aid of a literature review on known cellular changes during differentiation. The preparatory conditions were dropped, as they were fairly consistent across samples. A second selection round happened to eliminate variables which are similar representations of changes in cell shape, size and metabolism status in order to reduce variance. 

#### The most reliable model was the Random Forest Classifier. It performed the best. We initially attempted to structure the problem as a regression analysis and train a model accordingly, followed by a classification of the predicted dd10 CM content. This method produced reliable true negatives, but was very vulnerable to false negatives. As a result, we changed our approach to a classification method from the start. We initially transformed the y variable into a binary one, essentially assigning it a 1 if it had sufficient CM content (higher or equal to 90%). We trained the claasification model and used it to classify the test data. This method produced more consistent true positive results, leading to an increase in model performance according to given metrics, namely accuracy, precision, and recall. 

## Technologies Used
* python
* pandas
* sklearn





