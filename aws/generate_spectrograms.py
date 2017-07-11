# coding: utf-8
import pickle
import sys
import pandas as pd
import requests
import sox
import numpy as np
import os
from subprocess import run, PIPE
from PIL import Image
import tempfile
import re
import time
import string
from boto.s3.connection import S3Connection
from boto.s3.key import Key
import glob

conn = S3Connection('', '')
s3bucket = conn.get_bucket('spectrograms')
k = Key(s3bucket)

df = pd.read_pickle('final_data.pkl')

DEFAULT_IMG_SIZE = 256
DATA_DIR = ''

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
        img_name = '{}__{}.png'.format(input_file_cleaned, counter_formatted)
        start_width = dim[0]
        end_width = dim[1]
        sliced_img = img.crop((start_width, 0, end_width, DEFAULT_IMG_SIZE))
        sliced_img.save(DATA_DIR + img_name)
        counter += 1
    delete_file(input_file)

def create_file_names(id):
    genre_list = list(df['parent_genre'])
    genre_name = str(genre_list[id]).lower()
    genre_name = genre_name.replace('/','_')
    genre_name = genre_name.replace(' ','_')
    genre_name = genre_name.replace('&', 'n')
    id_list = list(df['id'])
    id_name = (id_list[id])
    track_name = '{}__{}.mp3'.format(genre_name, id_name)
    spect_name = track_name.replace('.mp3','')
    spect_name = '{}.png'.format(spect_name)
    return track_name, spect_name, genre_name

url_list = list(df['track_url'])
for track_id in range(len(df)):
    url = url_list[track_id]
    track_name, spect_name, genre_name = create_file_names(track_id)
    print('Track: {}, Spect: {}, Genre: {}'
          .format(track_name, spect_name, genre_name))

    try:
        download(url, track_name)
        set_to_mono(track_name)
        audio_to_spect(track_name, spect_name)
        slice_spect(spect_name)
        
        # all png files should now be in the working directory
        file_list = glob.glob('*.png')
        for file in file_list:
            # get genre from start of file name
            genre_name = file.split('__')[0]
            
            # set file name ready to upload to s3
            full_key_name = '{}/{}'.format(genre_name, file)
        
            try:
                # send file to s3
                k.key = full_key_name
                k.set_contents_from_filename(file)
                # once copied, delete from local
                delete_file(file)
            except:
                print('Problem copying file {}'.format(file))
                pass

        time.sleep(5)
        
    except KeyboardInterrupt:
        sys.exit()
    except:
        print('Something went wrong. Moving to next file')
        pass

