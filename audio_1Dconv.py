import os
import numpy as np
from scipy.io import wavfile
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint

### CREDITS - uses code from the following sources
# https://github.com/pkmital/CADL/blob/master/session-3/session-3.ipynb
# http://machinelearningmastery.com/predict-sentiment-movie-review

### STEP 1: EXTRACT DATA SPEECH AND MUSIC DATA FROM DATASET
# Get the full path to the directory
music_dir = os.path.join(os.path.join('./music_speech'), 'music_wav')

# Now use list comprehension to combine the path of the directory with any wave files
music = [os.path.join(music_dir, file_i)
         for file_i in os.listdir(music_dir)
         if file_i.endswith('.wav')]

# Similarly, for the speech folder:
speech_dir = os.path.join(os.path.join('./music_speech'), 'speech_wav')
speech = [os.path.join(speech_dir, file_i)
          for file_i in os.listdir(speech_dir)
          if file_i.endswith('.wav')]

# Let's see all the file names
print("Number of music files:", len(music), "| Number of speech files:", len(speech))

### Define a function for reading audio files (Parag Mital, Kadenze Online Class)
def load_audio(filename, b_normalize=True):
    """Load the audiofile at the provided filename using scipy.io.wavfile.

    Optionally normalizes the audio to the maximum value.

    Parameters
    ----------
    filename : str
        File to load.
    b_normalize : bool, optional
        Normalize to the maximum value.
    """
    sr, s = wavfile.read(filename)
    if b_normalize:
        s = s.astype(np.float32)
        s = (s / np.max(np.abs(s)))
        s -= np.mean(s)
    return s


# Read an audio file and check its length
file_i = music[0]
s = load_audio(file_i)
print("Length of an audio file:",len(s))  # This number is 661500 because audio is sampled at 22050Hz and samples are 30s long
print("Max amplitude:", np.max(s), "Min amplitude:", np.min(s))

### STEP 2: PREPARE THE DATASET
Xs = []

for i in range(len(music)):
    music_file = music[i]
    music_audio = load_audio(music_file)
    last_sample = int(len(music_audio)/30)  # I divide by 30 to get the index of a sample that is 1 s long. I discard the rest off the audio sample.
    music_audio = music_audio[0:last_sample]
    Xs.append(music_audio)

    speech_file = speech[i]
    speech_audio = load_audio(speech_file)
    speech_audio = speech_audio[0:last_sample]
    Xs.append(speech_audio)

Xs = np.array(Xs)
ys = np.hstack((np.ones(len(music)), np.zeros(len(speech))))

print("Xs.shape: ", Xs.shape, "ys.shape:", ys.shape)

### STEP3: SPLITTING and SHUFFLING THE DATASET into TRAINING AND VALIDATION SET
# Read this https://keras.io/getting-started/faq/#how-is-the-validation-split-computed
# You can set keras to automatically split the data into training and validation set. However
# Keras literally takes the last 10% of the data as validation, so we need to shuffle it beforehand.
# We do not however, need to generate a validation dataset and training set ourselves, which is convenient
# for prototyping.
# Note that Keras automatically shuffles the data when it trains (random minibatches -> that's why it's called
# Stochastic Gradient Descent (SGD)

# print (ys)
indices = np.random.permutation(Xs.shape[0])
Xs, ys = Xs[indices], ys[indices]
# print (ys)
Xs = np.reshape(Xs, (Xs.shape[0],Xs.shape[1],1))  # Conv1D expects a 3-dim array

### STEP 4: CREATE THE MODEL
model = Sequential()
model.add(Conv1D(input_shape=(Xs.shape[1],1),  # Length of input vector is contained in the 2nd term of np.shape function
                 filters=8,
                 kernel_size=4410,
                 padding='valid',  # Perhaps try 'causal' according to https://keras.io/layers/convolutional/#conv1d
                 activation='tanh'  # tanh goes from -1 to 1, it's basically a "better sigmoid"
                 ))
'''
model.add(Conv1D(filters=32,
                 kernel_size=441,
                 padding='valid',
                 activation='tanh'
                 ))
'''
model.add(Flatten())
model.add(LeakyReLU(512))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', validation_split=0.1, optimizer='adam', metrics=['accuracy'])
print(model.summary())

### STEP 5: BEGIN TRAINING THE MODEL
# Set up Keras checkpoints to monitor the accuracy and save the model when it improves
filepath="audio_kernels-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# This function actually starts the training
model.fit(Xs, ys, epochs=4, batch_size=16, callbacks=callbacks_list, verbose=2)

# Evaluate the model on the dataset
scores = model.evaluate(Xs, ys, verbose=0)

