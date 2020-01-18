# American_sign_language_understanding_Using_Machine_learning
Deaf is a disability that impairs the hearing, while mute is a disability that impairs the speaking. For the people affected by one of these disabilities the only way to communicate with others is by using sign language, where each gesture has a
specific meaning.
Still the general public doesn’t give the necessary impor- tance in learning Sign Language. One solution to tackle the communication gap is by using the service of sign language interpreter, however it’s very costly since people with this training are not easily found. So a cheap and easy solution is required so that deaf-mute and normal people can commu- nicate normally.
Nowadays, researchers try to solve the problem as cheaply and as efficiently as possible. The breakthrough for this problem is the Sign Language Recognition System. This system aims to recognize sign language, and translate it to the local language via text or speech. However, building this system is very expensive and is difficult to for daily use. Another advancement was the use of classification algorithms using machine learning. This was used on pictures or real time images. However, it still has some shortcomings, especially in the tracking of hand movements.
The problem of developing sign language recognition ranges from the image to the classification process. Researchers are still trying to find the best method for the images. Gathering images using a camera gives challenges in regards to image pre-processing. Classification methods also give researches some drawbacks, by having to recognize that there are several sign languages.
This report aims to discuss the Sign Language Recognition using Machine Learning Classification algorithms. This report explains the data set used and various Machine Learning algorithms used for Sign Language recognition.

# DATA DESCRIPTION
The American sign language data set used is from a GitHub or kerala database.
This data set is comprised in several images from A to Y. The letters J and Z aren’t in this data set because they have motion. Every letter is represented with 240 to 250 images. In total, the data set has an average of 5832 images.
These images were taken with a smartphone camera, some images represent some problems purposely because they are cross validation images. This problems are:
* Quality of images: some images are pixelated or poor on light;
* Aren’t centered: some images are on the left or right side;
* Images rotated: some are more rotated that others.

In this data set, 30% of the images of each letter are for testing, and 70% are for training.

In order to reduce the impact of processing the dataset raw, the following pre-processing steps were applied to each image:
* image resize
* grayscaling
These operations are equally applied to the test images.

# ALGORITHMS USED
Classification machine learning techniques like SVM, Neural Networks and Logistic Regression are used for supervised learning, which involves labeling the data set before feeding it into the algorithm for training. Feature extraction algorithms are used for dimensionality reduction in order to create a subset of the initial features such that only important data is passed to the algorithm. Grayscaling and image resize were used for this purpose.

## Support Vector Machine
Support vector machines is a supervised learning method with learning algorithms that analyze data used for classification, regression analysis and outliers detection.
In SVM, each data point is plotted in an n-dimensional space (n is the number of features) with the value of each feature being the value of a particular coordinate. The classification is done by finding a hyperplane that better differentiates the classes. A hyperplane is a decision plane which separates a set of objects having different class memberships.
The advantages of SVM are:
* Effective in high dimensional spaces.
* Still effective in cases where number of dimensions is greater than the number of samples.
* Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
* Versatile: different Kernel functions can be specified for the decision function.

The disadvantages of SVM include:
* If the number of features is much greater than the num- ber of samples, avoiding over-fitting in choosing Kernel functions and the regularization term is crucial.
* SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross- validation.
For this project a python version of SVM was created and uses the sklearn library (in use is sklearn.svm). SVC has the kernel set to Linear. A kernel transforms an input data space into the required form. A linear kernel can be used as a normal dot product.
The implementation is based on libsvm and multi-class support is handled according to a one-vs-one scheme. The objective of a Linear SVC (Support Vector Classifier) is to fit to the data provided, returning a best fit hyperplane that divides, or categorizes, the data. From here, after getting the hyperplane, it’s important then feed some features to the classifier to see what the predicted class is.
The implementation was done as follows:
* Load Data: First the data set was loaded with all the transformation on each image.
* Splitting Data: It was necessary to divide the data set into a training set and a test set. The separation of the dataset was done by the function train test split on model selection library from sklearn. The dataset is broken down in two parts with a ratio of 70:30. It means 70% of the data will be used for model training and 30% for model testing.
* Generating Model: The SVM module was imported and the support vector classifier object was created by passing the argument kernel as the linear kernel in SVC() func- tion. Then, the model was fited on train set and performed prediction on the test set.
* Evaluating the Model: The accuracy was computed by comparing the actual test set value and predicted value Figure 1.
In the end, was calculated accuracy of the model, Precision from each letter, Recall and F1-score. With this implemen- tation, the accuracy of this model is 96,29%. The Precision, Recall and F1-score are in the Table I.
It was also calculated the Micro Average, Macro Average and Weighted Average. So, Micro Average is a method that sums up the individual true positives, false positives and false negatives of the model for different sets and then applies them to get the statistics. In this case the result for precision, recall and f1 score in Micro average is 99%. Macro Average method take the average of the precision and recall of the system. In this case the result is equally 99%. Weighted Average is a method that calculate metrics for each label and find their average weighted by support (the number of true instances for each label). In this case the result for precision,recall and f1 score in Weighted average is 99%.

## Logistic Regression
Logistic Regression is one of the most simple and commonly used Machine Learning algorithms for two-class classification. It is easy to implement and can be used as the baseline for any binary classification problem. Its basic fundamental concepts are also constructive in deep learning. Logistic regression describes and estimates the relationship between one dependent binary variable and independent variables.
The advantages of Logistic Regression are:
* Efficient and Straightforward nature.
* Doesn’t require high computation power.
* Easy to implement.
* Easily interpretable.
* Used widely by data analyst and scientist.
* Doesn’t require scaling of features.
The disadvantages of Logistic Regression include:
* Is not able to handle a large number of categorical features/variables.
* It is vulnerable to overfitting.
* Can’t solve the non-linear problem.
* Will not perform well with independent variables that
are not correlated to the target variable and are very similar or correlated to eeach other.
So, in the multiclasse case, we set the parameter multi class on Logistic Regression function sklearn from linear model to multinomial. This means that the training algorithm uses the cross- entropy loss. This class, also, implements regularized regression using the ’liblinear’ library with ’lbfgs’ solvers. L-BFGS is an optimization algorithm in the fam- ily of quasi-Newton method that approximates the Broyden- Fletcher-Goldfarb-Shanno(BFGS) algorithm using a limited amount of computer memory. It is a popular algorithm for parameter estimation in machine learning. The algorithm finds a local minimum of an objective function, making use of objective function values and the gradient of the objective function. That level of description covers many optimization methods.
The sigmoid function, also called logistic function gives an S shaped curve that can take any real-valued number and map it into a value between 0 and 1. If the curve goes to positive infinity, y predicted will become 1, and if the curve goes to negative infinity, y predicted will become 0. If the output of the sigmoid function is more than 0.5, we can classify the outcome as 1 or YES,and if it is less than 0.5, we can classify it as 0 or NO.
The implementation is very similar to SVM algorithm. The following are the implementation steps:
* Load Data: First of all, it was loaded the data set with all the transformation on each image.
* Splitting Data: It was necessary dividing the data set into
a training set and a test set. Split the data set by using the function train test split on model selection library from sklearn (same as SVM model). The dataset is broken into two part in a ratio of 70:30. It means 70% data will be used for model training and 30% for model testing.
* Model Development and Prediction: Imported the Logis- tic Regression module and created a Logistic Regression classifier object using LogistiRegression() function from sklearn. Then, the model was fited on the train set using fit() and perorm prediction on the test set using predict().
* Model Evaluation using Confusion Matrix: A confusion matrix is a table that is used to evaluate the performance of a classification model. It also allows visualize the performance of an algorithm. The fundamental of a confusion matrix is the number of correct and incorrect predictions are summed up class-wise.
* Visualizing Confusion Matrix using Heatmap: The result of the model in the form of a confusion matrix using matplotlib and seaborn.
In the end, was calculated accuracy of the model, Precision from each letter, Recall and F1-score. With this implementation the accuracy of this model is 91,37%.

## Artificial Neural Network
Artificial Neural Network is a Machine Learning paradigm that tries to replicate the data processing that’s performed by the brain. A Neural Network architecture is commonly divided into 3 parts:
* Input Layer, which receives the training and testing data for processing
* Hidden Layer, which is comprised of 1 or more layers
* Output Layer, which the number of nodes depends on the Machine Learning problem
The classification problem being analysed in this research involves the distinction between 24 different types of images, so the number of output nodes must be 24.
The number of Input Layer nodes corresponds to the size of the images in the dataset. Originally it was decided to pre-process the dataset to be of size 30x30 pixels. Even though this resulted in faster processing, important details of the images were removed, which could result in less accurate classifications for images not in the dataset. To avoid this problem, the 100x100 pixels was used instead.
The image data was normalized and fed to the neural network for training.
#### Implementation
The neural networks training and testing was done with the class Matlab code as a base. In order to the image dataset in matlab, it was necessary to upload and convert all the images into a matrix X and create a matrix y. The matrix X has an image for each of its rows, and each columns corresponds to one pixel. Before create the matrix X, the images needed to be rotated in order to be visualized properly with the function for displaying one hundred images.
The activation function used was the one from the class, the Sigmoid Activation.
All the comparison tests made had, when feasible, the same initial weights and all the parameters the same except the one being tested.
Note: By distraction I forgot to split the dataset matrices X and y in two parts (70:30), so the tests where the global accuracy was measured used the dataset that was used to train the neural network.
#### Mini-Batch Learning
The training dataset used has almost 5000 examples. To improve the training performance of the neural network, the mini-batch learning process was used. To decide on the size to use for each batch, 3 alternatives were tested:
* Batch of 32 examples
* Batch of 128 examples
* Batch of 256 examples

Note: To compare the impact of the batch sizes, the same initial weights were used. In this graph each epoch correspond to the optimization of one batch of the dataset. The batch that resulted in the greatest accuracy after 2000 epochs was the batch with 128 examples.
It can also be concluded that the more times the entire dataset is passed through the optimization step, the more accurate the results will be. In the case of the 32 examples batch, in 2000 epochs, the entire dataset only passes 12 times through optimization. In the case of the 128 examples, the entire datasets passes 48 time and for the 256 examples batches it passes 96 times. However there’s a limit which can be seen for the 256 examples batch. If the size of a batch surpass a certain threshold, the learning performance of the NN will begin to suffer.

#### Dataset Organization
In batch learning the impact of the dataset organization is less pronounced because the entire dataset is optimized at the same time, at each optimization iteration, but that’s not the case for mini-batch learning. The following graph plots the evolution of the accuracy (each epoch correspond to the optimization of one batch of the dataset) through 2000 epochs for a dataset which has the the images of the same letters all following each other and a dataset with the dataset randomized.

It can be seen that the NN learning rate is stuck for the non-randomized dataset. This is because the NN optimizes the network in one iteration for one type of image. In the next iteration a new type of image may be fed to the optimiza- tion algorithm, which will drastically affect the cost and the accuracy and the network will optimize only for this new type.
By randomizing the entries of the dataset, it becomes more homogeneous. This proves to be effective due to various types of images being fed, in one batch, into the optimization algorithm. This allows for the optimization of the network for various types of images at each optimization iteration.
#### Neural Network regularization
To understand how regularization affects the performance of the neural network, various training simulations were made with different lambda values. In the graph of Figure 7 it can be seen that regularization has a relatively big impact on the resulted accuracy of the network, where the biggest accuracy was around 62% and the smallest was around 52%.
#### Number of Hidden Layer nodes
To test the impact of the number of hidden layer nodes on the learning rate of the network, for dimensions were tested: 25, 50, 100 and 400 nodes. 
The obtained accuracy for the entire dataset of each simu- lation can be seen in the following table:
So even though the accuracy of the hidden layer increases significantly as the number of hidden layer nodes increases, at each step the number of nodes necessary for a significant improvement also increases.
