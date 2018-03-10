### **Neural nets in simple terms**

* Start at some random set of weights
* Use for forward propogation to make prediction
* Use *backward progragation* to calculate the slope of the loss function w.r.t each weight
* Multiply that slope by the *learning rate*, and subtract it from the current weights
* Keep going with that cycle until we get to a flat part
 
#### **How backward propogation works in simple terms**

* Go back one layer at a time
* Gradients for weight is product of:
	1. Node value feeding into that weight (tail of the arrow)
	2. Slope of loss function w.r.t node it feeds into
	3. Slope of activation function at the node it feeds into ( in case of ReLu activation this is 1 for any node receiving a positive value and 0 otherwise)

#### **Stochastic gradient descent: for computational efficieny**
* Calculate slopes on only a subset of the data ("batch")
* Use a different batch of data to calculate the next update
* Start over from the beginning once all data is used
* Each time through training data is called an epoch
* In normal gradient descent, slopes are calculated using full data 

#### Keras and Tensorflow
**Keras** is a high-level neural networks API, written in python capable of running on top of **Tensorflow**, **CNTK**, **Theano**.

Model building steps :

* Specify architecture
* Compile
* Fit
* Predict

**Specifying a model**

```{python}
import keras
from keras.layers import Dense
from keras.models import Sequential

#number of columns in a predictors 
n_cols=predictors.shape[1] 

#Set up the model: model
model=Sequential()

# Add the first layer, say 50 neurons
model.add(Dense(50,activation="relu",input_shape=(n_cols,)))

#Add the second layer, say 32 neurons
model.add(Dense(32,activation="relu"))

#Add the output layer
model.add(Dense(1))

```
#### Compiling the model
We need to mention the optimizer and loss function. 
```{python}
model.compile(optimizer="adam",loss="mean_squared_error")
```
#### Fitting the model 
Scale your data(normalize) before fitting as it will be helpful in optimization. Apply backpropogation and gradient descent with your data to update weights
```{python}
model.fit(predictors,target)
```
Most commonly used loss function for classification problems is `categorical_crossentropy`/`log-loss` 
```{python}
model.compile(optimizer="sgd",loss="categorical_crossentropy",metrics=["accuracy"])
```
`sgd- stochastic gradient descent

#### Saving, reloading and using your model
```{python}
from keras.models import load_model
model.save('model_fil.h5')
my_model=load_model('my_model.h5')
predictions=my_model.predict(data_to_predict_with)
probability_true=predictions[:,1]
```

#### Fine tuning models with Keras
Iterate through multiple learning rates and use SGD (stochastic gradient descent) to arrive at an optimal value

**Dying neuron problem** - A "dead" neuron always outputs the same value (zero) for any input. Probably this is arrived at by learning a large negative bias term for its weights. Once a ReLu ends up in this state, it is unlikely to recover, because the function gradient at 0 is also 0, so gradient descent learning will not alter the weights. The sigmoid and tanh neurons can suffer from similar problems as their values saturate, but there is always at least a small gradient allowing them to recover in the long term.
 
**Vanishing gradients** 

Vanishing Gradient Problem is a difficulty found in training certain Artificial Neural Networks with gradient based methods (e.g Back Propagation). In particular, this problem makes it really hard to learn and tune the parameters of the earlier layers in the network. This problem becomes worse as the number of layers in the architecture increases.

This is not a fundamental problem with neural networks - it's a problem with gradient based learning methods caused by certain activation functions. 

*Problem*

Gradient based methods learn a parameter's value by understanding how a small change in the parameter's value will affect the network's output. If a change in the parameter's value causes very small change in the network's output - the network just can't learn the parameter effectively, which is a problem.

This is exactly what's happening in the vanishing gradient problem -- the gradients of the network's output with respect to the parameters in the early layers become extremely small. That's a fancy way of saying that even a large change in the value of parameters for the early layers won't have a big effect on the output.

*Cause*

Vanishing gradient problem depends on the choice of the activation function. Many common activation functions (e.g sigmoid or tanh) 'squash' their input into a very small output range in a very non-linear fashion. For example, sigmoid maps the real number line onto a "small" range of [0, 1]. As a result, there are large regions of the input space which are mapped to an extremely small range. In these regions of the input space, even a large change in the input will produce a small change in the output - hence the gradient is small.

This becomes much worse when we stack multiple layers of such non-linearities on top of each other. For instance, first layer will map a large input region to a smaller output region, which will be mapped to an even smaller region by the second layer, which will be mapped to an even smaller region by the third layer and so on. As a result, even a large change in the parameters of the first layer doesn't change the output much.

We can avoid this problem by using activation functions which don't have this property of 'squashing' the input space into a small region. A popular choice is Rectified Linear Unit which maps x to max(0,x)


```{python}
from keras.optimizers import SGD

model.compile(optimizer=SGD(lr=0.001),loss="categorical_crossentropy")

```

**Validation** - k-fold cross validation would not be used in deep learning as it would computationally very expensive
```{python}
model.compile(optimizer="adam",loss="categorial_crossentropy",metrics=["accuracy"])
model.fit(predictors,target,validation_split=0.3)
```
We need to keep training as the validation scores are improving and stop if they are not. This is called **Early stopping**

```{python}
from keras.callbacks import EarlySopping
early_stopping_monitor=EarlyStopping(patience=2)
model.fit(predictors,target,validation_split=0.3,epochs=20,callbacks=[early_stopping_monitor])

```
Possible experimentations -
 
* Different architectures
* More layers
* Fewer layers
* Layers with more nodes
* Layers with fewer nodes

Workflow for optimizing model capacity -

* Start with a small network
* Get the validation score
* Keep increasing the capacity until the validation score is no lnger improving

keras.io has an excellent document for referenece