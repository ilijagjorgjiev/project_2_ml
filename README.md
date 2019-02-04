# Road Segmentation Project, Machine Learning 2018 EPFL
## I. Structure of Folder

	1. src Folder => code

	2. model Folder => contains the model that is going to be saved each time the network is trained. It will be saved under a name new_model.h5.
											Also the model folder contains the best models that we got under the name best_model.h5

	3. submissions Folder => it will contain the generated submission ready for submission

	4. data Folder => should contain training_set and testing_set. That is how our run.py is structure, though you can change this parameters.
										The training_set is expected to have 2 other folders: images and groundtruth.
										Our code also expects to find all test images directly in the testing_set folder, instead of each image having its own folder within test_set

	5. predictions => this is where we save the predictions for each testing image, under a specified name and then we feed it to the masks_to_submission, so that it creates the submission file.

## II. File Organization of Src

Our code submission contains 5 different files
	- helpers_submissions.py : contains all the functions used to create submissions from the model
	- loss_and_metrics.py    : contains the custom loss function and F1_score metric used to train the model
	- unet.py                : contains the definition of our Neural Network's structure
	- read_augment_data.py   : contains the set of methods and objects used to augment our original training dataset
	- run.py                 : creates our Neural Network, trains it, and predicts with respect to the testing dataset

Each file contains comments explaining the different functions used as well as what parameters each one expects.

## III. Ranking

1. Got an f1 score of 0.902
2. Submission ID: 25104, Participant Name: Ilija Gjorgjiev, Team: Ilija Gjorgjiev, Adnan Ibrić, Raphael Laporte


## IV. How to run

For running just do python3 run.py

The run.py file by default is made to build a new model and train it again. Once the training is done, it will do all the predictions and save it to the ./submissions/best_submission.csv

If you would like to use our best model, which can be found in ./model/best_model.h5
You just need to modify the variable trainModel to False, thus will load the best model and predict based on it.

## V. GPU used

RTX 2080 8GB system

For recreating our best model, we used a model with 12500 epochs, with 5 steps_per_epoch. It took us around 7 hours to get it trained to get 0.902 f1 score.

## VI. Package Versionning

The version of TensorFlow used is 1.11.0.
The version of Keras used is  2.2.4
