# The data set is taken from Kaggle.com
# This data set is a handwritten digits image data set. Every digit is represented by a 28x28 image.
# First the data set is converted into table using read.csv() function. 

train <- read.csv("train.csv",header = TRUE)

# Now let us break our data set into 2 parts training and testing use createDataPartition() function which is available in caret package

intrain <- createDataPartition(y = train$label,p=0.8,list = FALSE)

# in above command p=0.8 means we are breaking the data set in the ratio of 80:20

newtrain <- train[intrain,]

newtest <- train[-intrain,]

# newtrain will be 80% of our data and newtest will be 20%
# newtrain will be used only for model fitting and newtest will be used for prediction

train <- data.matrix(newtrain)
test <- data.matrix(newtest)

# We converted both the data frames into matix, because for applying Convolutional Neural Network(CNN), input should be in the form of matrix

train.x <- train[,-1]
train.y <- train[,1]

# train.x will contain all the predictor i.e independent variables and train.y will contain the outcome variable which is nothing but the label of image

train.x <- t(train.x/255)

# In above command we have scaled the values of pixel from 0 to 1.
# Since all the images were greyscaled , so all the pixel values were between 0 and 255
# For scaling between 0 to 1 , formula is (x[i]-min(x))/(max(x)-min(x))
# So for our case it simply become x[i]/255

# Now doing same for test matrix

test.x <- test[,-1]
test.y <- test[,1]
test.x <- t(test.x/255)

table(train.y)
table(test.y)

# In above command it shows how many 0's are there,how many 1's,...,how many 9's are there in both train and test matrix

# Now let us first make the CNN model
# First load the mxnet package

data <- mx.symbol.Variable('data')

# Above command simply create a symbolic variable with the name 'data'
 
# Now let us create first convolution layer

conv_1 <- mx.symbol.Convolution(data = data,kernel = c(3,3),num.filter = 32)

# In above command we have applied 32 filters of size 3X3 to our symbol input data 
# The output of above command will be used as input to next used function

rect_1 <- mx.symbol.Activation(data = conv_1,act.type = "relu") 

# Above command applies activation function to the data we got from our convolution layer
# Activation function is applied to increase non-linearity in our images because images themselves are highly non-linear and when we apply filters, we risk of making it linear
# Activation function used here is Rectified Linear Unit(relu) which is f(x) = max(x,0)

pool_1 <- mx.symbol.Pooling(data = rect_1,pool.type = "max",kernel = c(2,2),stride = c(2,2))

# In above command we have performed pooling
# pooling type used is 'max' pooling of size 2X2 and shift window of 2X2

# Another convolution layer can also be added here

# Now we will flatten the input data into a 2-D array by collapsing the higher dimensions

flat <- mx.symbol.flatten(data = pool_1)

# Now this flattened data will be sent to the fully coonected layer

fc1 <- mx.symbol.FullyConnected(data = flat,num.hidden = 128)

# In above command num.hidden = 128 means number of neurons or nodes are 128
# Now we will again apply activation function on the new output

rect_3 <- mx.symbol.Activation(data = fc1,act.type = "relu") 


fc2 <- mx.symbol.FullyConnected(data=rect_3, num_hidden=10)

# In above command we are again creating a fully connected layer but this time number of hidden nodes are specified as 10 as we need output ranging from 0 to 9

lenet <- mx.symbol.SoftmaxOutput(data=fc2)

# Above command computes the gradient of cross entropy loss with respect to softmax output

train.array <- train.x
dim(train.array) <- c(28, 28, 1, ncol(train.x))

# In above command we are explicitly specifying the dimensions as 28X28 as all the images are of size 28X28, and here 1 denotes that the images are greyscaled, if they would have been coloured then it would have been 3

test.array <- test.x
dim(test.array) <- c(28, 28, 1, ncol(test.x))
 
# Now let us train the model on the data
mx.set.seed(0)

# Above command set the seed used by mxnet device-specific rendom number generators, so that when model runs for several time output does not change drastically

devices <- mx.cpu()

# Above command create a mxnet CPU context

model <- mx.model.FeedForward.create(lenet, X=train.array, y=train.y,
                                     ctx=devices, num.round=5, array.batch.size=100,
                                     learning.rate=0.05, momentum=0.9, wd=0.00001,
                                     eval.metric=mx.metric.accuracy,
                                     epoch.end.callback=mx.callback.log.train.metric(100))

# Above command create a MXNet feedforward neural net model
# lenet is the symbolic configuration of the neural network that we have designed
# ctx tell which device is used to perform training
# num.round tells the number of iterations to be done over training data to train the model
# eval.metric tell the evaluation function to be done on results, we have used accuracy function


# Now we will predict our model on test data

predicted_probs <- predict(model,test.array)

# You will see that the predicted_probs contain 10 probability values for each image , the one with the highest predicted value is the label of that image

predicted_labels <- max.col(t(predicted_probs)) - 1

# Here we have used max() function to only use the highest predicted value , -1 is done because R has given us range from 1 to 10 and we want range from 0 to 9 as our labels

table(test[,1], predicted_labels)

# From above command you can see how much we predicted right.

