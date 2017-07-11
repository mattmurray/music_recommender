# coding: utf-8
import matplotlib
from keras import backend as K
import requests
import os
import eyed3
from subprocess import run, PIPE
from PIL import Image
from keras.applications.vgg16 import preprocess_input
import glob
from tkinter.filedialog import askopenfilename
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Input, Dense
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.optimizers import rmsprop

# Set variables
image_size = 256
num_classes = 9
DEFAULT_IMG_SIZE = 256
DATA_DIR = 'prediction/'

class_labels = {
    0:'Breakbeat',
    1:'Dancehall/Ragga',
    2:'Downtempo',
    3:'Drum and Bass',
    4:'Funky House',
    5:'Hip Hop/R&B',
    6:'Minimal House',
    7:'Rock/Indie',
    8:'Trance'
}

# Get filename input
filename = askopenfilename()
track_name = os.path.basename(filename)
print('Analyzing {}...'.format(track_name))
spect_name = 'spect.png'

### Functions
def get_genre(s):
    genre = s.split('/')[0]
    return genre

def get_song_id(s):
    id = s.split('_')[1]
    return id

def get_spect_num(s):
    spect = s.split('_')[2]
    return spect

def get_class_name(s):
    s = int(s.split('_')[1])
    return y[s]

# downloads the mp3 from juno
def download(url, file_name):
    with open(file_name, "wb") as file:
        response = requests.get(url)
        file.write(response.content)

# helper function to delete files no longer needed
def delete_file(file_path):
    os.remove(file_path)

# creates a mono version of the file
# deletes original stero mp3 and renames the temp
# mono file to the original stero filename
def set_to_mono(input_file):
    tmp_name = 'tmp.mp3'
    command = "sox {} {} remix 1,2".format(input_file, tmp_name)
    run(command, shell=True, stdin=PIPE, stdout=PIPE)
    delete_file(input_file)
    os.rename(tmp_name, input_file)

# converts the audio to spectrogram
def audio_to_spect(input_file, output_file):
    command = "sox {} -n spectrogram -Y 300 -X 50 -m -r -o {}".format(input_file, output_file)
    run(command, shell=True, stdin=PIPE, stdout=PIPE)
    delete_file(input_file)

# helper function - gets dimensions of the spectrogram
def get_spect_dims(input_img):
    img_width, img_height = input_img.size
    return img_width, img_height

# helper function - calculates the number of slices available from the full size spectrogram
def get_num_slices(img_width):
    n_slices = img_width // DEFAULT_IMG_SIZE
    return n_slices

# helper function - returns a list of coordinates/dimensions where to split the spectrogram
def get_slice_dims(input_img):
    img_width, img_height = get_spect_dims(input_img)
    num_slices = get_num_slices(img_width)
    unused_size = img_width - (num_slices * DEFAULT_IMG_SIZE)
    start_px = 0 + unused_size
    image_dims = []
    for i in range(num_slices):
        img_width = DEFAULT_IMG_SIZE
        image_dims.append((start_px, start_px + DEFAULT_IMG_SIZE))
        start_px += DEFAULT_IMG_SIZE
    return image_dims

# slices the spectrogram into individual sample images
def slice_spect(input_file):
    input_file_cleaned = input_file.replace('.png','')
    img = Image.open(input_file)
    dims = get_slice_dims(img)
    counter = 0
    for dim in dims:
        counter_formatted = str(counter).zfill(3)
        img_name = '{}_{}.png'.format(input_file_cleaned, counter_formatted)
        start_width = dim[0]
        end_width = dim[1]
        sliced_img = img.crop((start_width, 0, end_width, DEFAULT_IMG_SIZE))
        sliced_img.save(DATA_DIR + img_name)
        counter += 1
    delete_file(input_file)

def create_file_names(id):
    genre_name = str(df['release_genre'][id]).lower()
    genre_name = genre_name.replace('/','_')
    genre_name = genre_name.replace(' ','_')
    id_name = (df['id'][id])
    track_name = '{}_{}.mp3'.format(genre_name, id_name)
    spect_name = track_name.replace('.mp3','')
    spect_name = '{}.png'.format(spect_name)
    return track_name, spect_name

def is_mono(filename):
    audiofile = eyed3.load(filename)
    return audiofile.info.mode == 'Mono'

if K.image_data_format() == 'channels_first':
    input_shape = (3, image_size, image_size)
else:
    input_shape = (image_size, image_size, 3)

# Load model
model = load_model('music_genre_classifier.hdf5')

# Create and save spectrograms
if is_mono(track_name) == False:
    set_to_mono(track_name)

audio_to_spect(track_name, 'spect.png')
slice_spect(spect_name)

# Predict against saved spectrograms
spect_files = glob.glob('prediction/*.png')

images = []
for file in spect_files:
    img = load_img('{}'.format(file), target_size=(256, 256))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255
    images.append(img_array)

print('Analyzing file...')

predictions = []
for image in images:
    prediction = model.predict(image)
    predictions.append(prediction)

# Sum individual probabilities for all spectrograms
pred_sum = sum(a[0] for a in predictions)
biggest_num = np.amax(pred_sum)
pct_confidence = round((biggest_num / sum(pred_sum) * 100), 2)
pred_class_num = np.argmax(pred_sum)

print('\nPrediction = {}, with {}% confidence. \n\n(This is based on a sum of probabilities from all 5(ish) second spectrograms created from the audio file).'.format(class_labels[pred_class_num], pct_confidence))