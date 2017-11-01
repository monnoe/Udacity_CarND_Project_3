import os
import csv
import random
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, Convolution2D, MaxPooling2D, Conv2D
from keras.optimizers import adam as adam1, adamax as adam2, SGD as sgd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle


def image_plot(image, name):
	#print(name)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	plt.close('all') 
	#print("printing Center Image data treatment!!! "+name)
	fig0 = plt.figure()
	plt.imshow(image)
	plt.title(name)
	plt.show()

def image_reader(line, img_coll, ang_coll):
	#path_name = '.\\data_udacity\\data\\IMG\\' #used on Udacity training data
	path_name = '.\\data_training_2\\' #used on Kamil's training data
	name = path_name + line[0].split('/')[-1] 
	name_left = path_name + line[1].split('/')[-1]
	name_right =  path_name + line[2].split('/')[-1]
	#print("Treating images : \n"+name + "\n"+name_left + "\n"+name_right + "\n")
	
	center_image = []
	left_image = []
	right_image = []
	center_image = cv2.imread(name)[80:160]
	left_image = cv2.imread(name_left)[80:160]
	right_image = cv2.imread(name_right)[80:160]
	center_angle = float(line[3])
	angle_correction = 0.2 #if an object is front and center of the center camera (0°), then the left camera sees the same object at an angle 0°+0.2° and the right camera angle sees it at 0° - 0.2°  
	left_angle = center_angle - angle_correction 
	right_angle = center_angle + angle_correction
	#image_plot(center_image, name)
	#print("angle :"+str(center_angle))
	img_coll.append(center_image)
	ang_coll.append(center_angle)
	img_coll.append(left_image)
	ang_coll.append(left_angle)
	img_coll.append(right_image)
	ang_coll.append(right_angle)
	
	if abs(center_angle) > 0.01: #new check : > 0.01: # old check: <= 10.00:  #attempt to get rid of outlier images numerically//  1.00 --> 10.00 just to capture the full dataset independantly
		flipped_center_image = []
		flipped_center_image = cv2.flip( center_image, 1 )
		flipped_center_angle = -1*center_angle
		flipped_left_image = []
		flipped_left_image = cv2.flip( left_image, 1 )
		flipped_left_angle = -1*left_angle
		flipped_right_image = []
		flipped_right_image = cv2.flip( right_image, 1 )
		flipped_right_angle = -1*right_angle
		#image_plot(flipped_center_image, name)
		#print("angle :"+str(flipped_center_angle))
		img_coll.append(flipped_center_image)
		ang_coll.append(flipped_center_angle)
		img_coll.append(flipped_left_image)
		ang_coll.append(flipped_left_angle)
		img_coll.append(flipped_right_image)
		ang_coll.append(flipped_right_angle)
		#noisy_steer_angle = center_angle + random.uniform(-0.1,0.1) #I suspect that this was polluting my data instead of augmenting it
		#can look at how to use right and left images here too...
		#img_coll.append(center_image)
		#ang_coll.append(noisy_steer_angle)
		

	#other attempts to creating random variations in the data:
	#m=(10,10,10)
	#s=(10,10,10)
	
	#noisy_center_image = []
	#noisy_center_image = cv2.randn(center_image,m,s)

	
	#noisy_flipped_center_image = []
	#noisy_flipped_center_image = cv2.randn(flipped_center_image,m,s)
	return img_coll, ang_coll

#image_reader(name)

def samples_creator(drive_log, samples):
	#samples = []
	with open(drive_log) as csvfile:
		next(csvfile, None)
		reader = csv.reader(csvfile)
		for line in reader:
			center_angle = float(line[3])
			#samples.append(line)
			#if (drive_log != './driving_log_Udacity.csv') and abs(center_angle) > 0.01:
			if abs(center_angle) >= 0.010:#// include everyone!
				samples.append(line)
			#if (drive_log =='./driving_log_Udacity.csv'):
			#	samples.append(line)
	print('samples size :' + str(len(samples)))
	return samples

def samples_processor(samples, images, angles):
	for line in samples:
		images, angles = image_reader(line, images, angles)
	return images, angles

#Paths to the driving logs and image collections: 
#drive_log_all = ['.\\data_udacity\\data\\driving_log.csv'] #--> good data to start with
drive_log_all = ['.\\data_training_2\\driving_log_new.csv'] #--> dataset from Kamil (who has successfully trained his model with it...)

samples = []

for drive_log in drive_log_all:
	print('running through samples in ' + drive_log)
	samples = samples_creator(drive_log, samples)
		
train_samples, test_samples = train_test_split(samples, test_size=0.25)

train_images = []
train_angles = []
test_images = []
test_angles = []

test_images, test_angles = samples_processor(test_samples, test_images, test_angles)
train_images, train_angles = samples_processor(train_samples, train_images, train_angles)
#print(test_angles)
#print(train_angles)

X_train = np.array(train_images)
y_train = np.array(train_angles)
X_test = np.array(test_images)
y_test = np.array(test_angles)

print("Shape training image samples : ", str(X_train.shape))
print("Shape validation image samples : ", str(X_test.shape))


#################################################################################

#model adapted from alemenis on github and Udacity video in the behavioral cloning project "10. More Networks"

# Parameters

nb_rows  = 80 # image size, see center_image for the number of rows to use
nb_cols  = 320
#steering_theta = 0.3 # side cameras correction angle
#test_samples = 20   # use few images to test the CNN
epochs = 30
batch_size = 1024
# Build the model
model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5,input_shape=(nb_rows, nb_cols, 3), name='Normalization'))
model.add(Conv2D(6, 5, 5, border_mode='same', activation='relu'))
model.add(MaxPooling2D((4, 4), (4, 4), 'same'))
model.add(Conv2D(6, 5, 5, border_mode='same', activation='relu'))
model.add(MaxPooling2D((4, 4), (4, 4), 'same'))
model.add(Conv2D(6, 5, 5, border_mode='same', activation='relu'))
model.add(MaxPooling2D((4, 4), (4, 4), 'same'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))  #  --> this guy here is supposed to give out a steering indication...

model.summary() # print model summary

# Training
#model = load_model('model.h5') #so the model doesn't have to learn from scratch, it uses the previous 
#####
##adam1 = adam1(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#adam1 = adam2(lr=0.0015, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001)
sgd1 = sgd(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',optimizer=sgd1,accuracy = 'accuracy')
#    
## train the model using the 10% of the training dataset for validation
#model.fit(X_train, y_train, batch_size=batch_size,nb_epoch=epochs, verbose=1, validation_split=0.1)
#
##del model  # deletes the existing model
#
## returns a compiled model
## identical to the previous one
##model = load_model('my_model.h5')
######

#model.compile(sgd', 'sparse_categorical_crossentropy', ['accuracy'])
history = model.fit(X_train, y_train, nb_epoch=epochs, validation_split=0.2)


# Save model architecture
print("Save model")
model_json = model.to_json()
import json
with open('model.json', 'w') as f:
    json.dump(model_json, f, ensure_ascii=False)
#model.save_weights("model.h5")
model.save("model.h5") # creates a HDF5 file 'model.h5'
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#


##
# Test run
P = model.predict(X_test[:])
#P = model.predict(X_valid[:])
P = P.reshape((P.shape[0],)) 

## plot predictions over line of equality
#sns.set_style("white")
fig = plt.figure(1, figsize=(5,5))
fig.clf()
ax = fig.add_subplot(111)

ax.scatter(P, y_test, marker='o', color="orchid", s=70, zorder=10)
#ax.scatter(P, y_valid, marker='o', color="orchid", s=70, zorder=10)
plt.plot([-0.5,0.5],[-0.5,0.5], 'k--', label="line of equality")
plt.xlabel("prediction")
plt.ylabel("y_test")
plt.legend(loc='best')
plt.tight_layout()
plt.draw()
plt.show()
##
