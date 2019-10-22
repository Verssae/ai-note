# AI Midterm Exam: Important Topics

* **Terminology of data matrix**: features, samples, etc.
  * Samples = instances = observations
  * Features = attributes = measurements = dimensions
* The concepts of **training, testing, validation sets**.
  * training: a dataset of samples used for learning, that is to fit the parameters of a classifier
  * validation: a dataset of samples used to tune the hyperparameters of a classifier and model selection.
  * test: a dataset independent of the training dataset, but follows the same probability distribution as the training set and used to evaluate the fitted model.
* Differences between **supervised and unsupervised learning**.
  * Supervised Learning: learn a model from labeled training data, that allows us to make predictions about unseen (future) data points
    * Classification: a supervised learning task with discrete class labels
    * Regression: a supervised learning task where outcome signal is a continuous value
  * Unsupervised learning: we deal with unlabeled data. We explore the structure of our data to extract mearningful information without the guidance of a known outcome varible or reward function.
    * Clustering: an exploratory data analysis technique that allows us to organize a pile of information into meaningful subgroups (clusters) without having any prior knowledge of their group memberships.


* The reason **why we may need scaling and centering (or, standardization)**

  * If a feature's variance is orders of magnitude more than the variance of other features, that particular feature might dominate other features in dataset, which is not  something we want happening in our model. And Many ML algorithms behave better when features are on the same scale. So we need scaling and centering.


* What is the **“no free lunch theorem”?**
   
* [link](https://ml-dnn.tistory.com/1)
   * an ml algorithm is designed to perform well on certain tasks, which requires certain assumptions
   * there is no universal ML algorithm that performs well on all of the tasks.
   * averaged over all the problems possible, the performance of all classifier is the same
   * therefore it is essential to compare a handful of different algorithms in order to train and select the best performing model 
   
* **Perceptron**: the ideas of...
   * Activation function : converts a linear prediction to a nonlinear output
   
   * Learning rate: how fast update weights called eta
   
   * Weight update: 
     $$
     w_j = w_j + \Delta w_j, \;\; \Delta w_j = \eta (y^{(i)} - \hat y^{(i)} ) x_j^{(i)}, \;\; j=1,2,\dots, m \quad \text{[Perceptron Learning Rule]}
     $$
   
   * Linear separability: in d-dimensional space, a set of points with
labels in {+, -} is linearly separable if there exists a hyperplane in the same space such that all the points labeled + lie to one side of the hyperplane, and all the points labeled - lie to the other side of the hyperplane.
   
   * Convergence: only guaranteed if
   
     * the two classes are linearly separable
     * the learning rate is sufficiently small.


* How can we do **multi-class classification via binary classification**
  * Binary classification: distinguish between two possible classes (e.g. spam and non-spam emails)
  * Multi-class classification: distinguish amongst multiple classes (e.g. handwritten digits from 0 to 9)
  * one  vs all : train one classifier per class, where the particular class is treated as the positive class, and all other classes are considered as the negative class and choose the label associated with the largest absolute net input value .


* **Adaline**:
   * Difference to perceptron: perceptron uses the class label to learn model coefficient. Adaline uses continuous predicted net output values, which is more powerful since it tells us by how much we were right or wrong. So in adaline, activation function is identical function. The weight update is calculated based on all training samples 
   * Gradient descent: update weight using gradient of cost function and learning rate eta.
   * Stochastic gradient descent (with/without minibatch): If sample is too many, compute all gradient is too time consuming. So stochastically select a single random sample and compute it to get gradient. Or using mini-batch, we select stochastically a small subsamples of the training data.


* **Logistic regression**
   * Logit function : 
   * Logistic sigmoid function
   * Likelihood function
   * Maximum likelihood estimation


* **SVM**
   * Margin
   * Maximum margin
   * The parameter C
   * Kernels: definition, rbf, poly


* **Model selection**
   * K-fold CV
   * Using validation set
   * Grid search


* **Python / scikit-learn**
   * How to run Perceptron, Logistic regression, and SVM in scikit-learn
   * Parameters (C, gamma, random_state, …)
   * Plotting data points
   * Plotting decision boundaries


* **PBL1**
   * Understanding of the MNIST data
   * How can we extract useful information from the data
   * Performance differences between Perceptron, Logistic regression, and SVM
   * Strategies for grid search (with time limits, for example)


* loss vs cost vs objective function : [link](https://blog.naver.com/PostView.nhn?blogId=qbxlvnf11&logNo=221386278997&parentCategoryNo=&categoryNo=&viewDate=&isShowPopularPosts=false&from=postView)