#!/usr/bin/env python3
# -*- coding = utf-8 -*-
from __future__ import absolute_import, unicode_literals

import os
import sys
import random
import argparse

import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm

import tensorflow as tf

np.random.seed(0)

__all__ = ['Dataset']

class Dataset(object):
   """
   Dataset class for the DAiSEE dataset.
   """
   def __init__(self, detector = 'dnn', **kwargs):
      # Initialize and validate all class arguments (set up class).
      self.detector_name = detector.lower().replace(" ", "")
      self._parse_kwargs(kwargs)
      self._initialize_detector()

      # Get dataset attributes.
      self.train_labels, self.validation_labels, self.test_labels = self._read_labels()

      # Get usable datasets.
      self._train_data = None
      self._validation_data = None
      self._test_data = None

   def __call__(self, detector = None, **kwargs):
      """Allow updates to class attributes."""
      self.detector_name = detector
      self._parse_kwargs(kwargs)
      self._initialize_detector()

   def __getattr__(self, item):
      """Account for any code mistakes."""
      if item == 'cascade_path' and self.detector != 'cascade':
         raise AttributeError("You are trying to get the cascade_path argument, but you are not using a cascade classifier.")
      if item in ['dnn_model_path', 'dnn_weights_path'] and self.detector != 'dnn':
         raise AttributeError("You are trying to get a DNN path argument, but you are not using the DNN detector.")
      else:
         raise AttributeError(f"No such attribute {item} in class Dataset.")

   def __contains__(self, item):
      """Determine whether video ID is in class."""
      return self._is_video(item)

   @property
   def train_data(self):
      """Training data accessor."""
      return self._train_data

   @train_data.setter
   def train_data(self, value):
      """Internal setter for training data."""
      self._train_data = value

   @property
   def validation_data(self):
      """Validation data accessor."""
      return self._validation_data

   @validation_data.setter
   def validation_data(self, value):
      """Internal setter for validation data"""
      self._validation_data = value

   @property
   def test_data(self):
      """Test data accessor."""
      return self._test_data

   @test_data.setter
   def test_data(self, value):
      """Internal setter for test data."""
      self._test_data = value

   def _parse_kwargs(self, kwargs) -> None:
      """Internal method to parse keyword arguments."""
      pot_kwarg = ['image_size', 'dataset_path', 'dnn_model_path' 'dnn_weights_path', 'cascade_path']
      if kwargs is None: # If no keyword arguments, then run with defaults.
         return None
      for kwarg in kwargs: # Validate keyword arguments.
         if kwarg not in pot_kwarg:
            raise ValueError(f"You have provided an invalid keyword argument: {kwarg}.")

      # Initialize keyword arguments in class instance.
      for _kwarg in kwargs:
         kwarg = _kwarg.lower().replace(" ", "")
         if kwarg == 'image_size': # Validate and set image size keyword argument.
            temp_image_size = kwargs['image_size']; _image_size = None
            if isinstance(temp_image_size, int):
               _image_size = (temp_image_size, temp_image_size)
            elif isinstance(temp_image_size, (list, tuple, set)):
               if len(temp_image_size) != 2:
                  raise ValueError("You must provide a two-dimensional argument for image size.")
               else:
                  _image_size = temp_image_size
            else:
               raise TypeError(f"You have provided an invalid argument for image_size of type {type(temp_image_size)}")
            setattr(self, 'image_size', _image_size)
            del temp_image_size, _image_size

         elif kwarg == 'dataset_path': # Validate and set dataset path keyword argument.
            temp_dataset_path = kwargs['dataset_path']; _dataset_path = None
            if not os.path.exists(temp_dataset_path):
               raise NotADirectoryError("You have provided an invalid dataset_path argument, as it does not exist.")
            if len(os.listdir(temp_dataset_path)) != 6:
               raise ValueError("The path you have provided for dataset_path does not contain files in the proper arrangement.")
            else:
               _dataset_path = temp_dataset_path
            setattr(self, 'dataset_path', _dataset_path)
            del temp_dataset_path, _dataset_path

         elif kwarg == 'cascade_path': # Validate and set cascade path keyword argument.
            if self.detector_name != 'cascade':
               raise ValueError("You have provided the cascade classifier path, but you are not using the cascade classifier.")
            temp_cascade_path = kwargs['cascade_path']; _cascade_path = None
            if not os.path.exists(temp_cascade_path):
               raise FileNotFoundError(f"The cascade classifier path {temp_cascade_path} does not exist.")
            else:
               _cascade_path = temp_cascade_path
            setattr(self, 'cascade_path', _cascade_path)
            del temp_cascade_path, _cascade_path

         elif kwarg in ['dnn_model_path', 'dnn_weights_path']: # Validate and set DNN path keyword arguments.
            if self.detector_name != 'dnn':
               raise ValueError("You have provided the DNN model/weights path, but you are not using the DNN detector.")
            if kwarg == 'dnn_model_path' and ('dnn_weights_path' not in kwargs):
               raise ValueError("If you want to use a custom DNN detector, you have to provide both a model and weights path.")
            elif kwarg == 'dnn_weights_path' and ('dnn_model_path' not in kwargs):
               raise ValueError("If you want to use a custom DNN detector, you have to provide both a model and weights path.")
            else:
               if hasattr(self, 'dnn_model_path') or hasattr(self, 'dnn_weights_path'):
                  continue
               else:
                  temp_dnn_model_path = kwargs['dnn_model_path']; _dnn_model_path = None
                  if not os.path.exists(temp_dnn_model_path):
                     raise FileNotFoundError("You have provided an invalid argument for the DNN model path.")
                  else:
                     _dnn_model_path = temp_dnn_model_path
                  temp_dnn_weights_path = kwargs['dnn_weights_path']; _dnn_weights_path = None
                  if not os.path.exists(temp_dnn_weights_path):
                     raise FileNotFoundError("You have provided an invalid argument for the DNN weights path.")
                  else:
                     _dnn_weights_path = temp_dnn_weights_path
                  setattr(self, 'dnn_model_path', _dnn_model_path)
                  setattr(self, 'dnn_weights_path', _dnn_weights_path)
               del temp_dnn_model_path, temp_dnn_weights_path, _dnn_model_path, _dnn_weights_path

      self._set_default_kwargs() # Set default keyword arguments if not provided.

   def _set_default_kwargs(self) -> None:
      """Internal method to set default keyword arguments."""
      if not hasattr(self, 'image_size'):
         setattr(self, 'image_size', (224, 224))
      if not hasattr(self, 'dataset_path'):
         _dataset_path = os.path.join(os.path.dirname(__file__), 'data/daisee-dataset')
         setattr(self, 'dataset_path', _dataset_path)
         del _dataset_path

   def _initialize_detector(self) -> None:
      """Internal method to initialize facial detector."""
      if self.detector_name == 'mtcnn': # Account for MTCNN.
         raise ValueError("The MTCNN detector is too computationally expensive for the amount of images being processed.")
      if self.detector_name not in ['dnn', 'cascade']: # Validate detector.
         raise ValueError(f"You have provided an invalid detector: {self.detector}. Valid detectors are [dnn, cascade].")

      # Initialize Cascade Classifier.
      if self.detector_name == 'cascade':
         if hasattr(self, 'cascade_path'): # If a cascade classifier path has been provided.
            face_cascade = cv2.CascadeClassifier(getattr(self, 'cascade_path'))
         else: # Otherwise, use the default (within-package) path.
            temp_path = os.path.join(os.path.dirname(cv2.__file__), 'data/haarcascade_frontalface_default.xml')
            try:
               face_cascade = cv2.CascadeClassifier(temp_path)
            except Exception as e:
               raise e
            finally:
               del temp_path

         # Add cascade classifier to class instance.
         setattr(self, 'cascade', face_cascade)

      # Initialize DNN Detector.
      if self.detector_name == 'dnn':
         if hasattr(self, 'dnn_model_path') and hasattr(self, 'dnn_weights_path'): # If DNN file paths have been provided.
            net = cv2.dnn.readNetFromCaffe(getattr(self, 'dnn_model_path'), getattr(self, 'dnn_weights_path'))
         else: # Otherwise, use the default path.
            temp_path = os.path.join(os.path.dirname(__file__), 'data/dnnfile')
            temp_model_path = os.path.join(temp_path, 'model.prototxt')
            temp_weights_path = os.path.join(temp_path, 'res10_300x300_ssd_iter_140000_fp16.caffemodel')
            try:
               net = cv2.dnn.readNetFromCaffe(temp_model_path, temp_weights_path)
            except Exception as e:
               raise e
            finally:
               del temp_path, temp_model_path, temp_weights_path

         # Add DNN detector to class instance.
         setattr(self, 'detector', net)

   @staticmethod
   def _get_images(datadir):
      """Internal method to get images from a tree folder."""
      if not os.path.exists(datadir): # Validate Directory.
         raise OSError(f"The path {datadir} does not exist.")

      dir_images = []
      for _user in os.listdir(datadir): # Get all images for a certain directory.
         if sys.platform == 'darwin' and _user == '.DS_Store':
            continue  # Need to watch out for .DS_Store
         for _video in os.listdir(os.path.join(datadir, _user)):
            if sys.platform == 'darwin' and _video == '.DS_Store':
               continue # Need to watch out for .DS_Store
            for _picture in random.sample(os.listdir(os.path.join(datadir, _user, _video)), 3):
               if _picture.endswith(".jpg") or _picture.endswith(".png"):
                  dir_images.append(os.path.join(datadir, _user, _video, _picture))

      return dir_images

   def _read_labels(self) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
      """Internal method to acquire dataset labels."""
      _label_loc = os.path.join(self.dataset_path, 'Labels')

      # Read label files and return label dataframe.
      try:
         train_df = pd.read_csv(os.path.join(_label_loc, 'TrainLabels.csv'))
         validation_df = pd.read_csv(os.path.join(_label_loc, 'ValidationLabels.csv'))
         test_df = pd.read_csv(os.path.join(_label_loc, 'TestLabels.csv'))
      except FileNotFoundError:
         raise FileNotFoundError("One or more of the label-containing files is missing.")
      except Exception as e:
         raise e
      finally:
         del _label_loc

      # Remove trailing space in attributes.
      for df in [train_df, validation_df, test_df]:
         df = df.rename(columns = {'Frustration ': 'Frustration'})

      return train_df, validation_df, test_df

   def _is_video(self, name):
      """Internal utility method to determine whether a certain video exists."""
      _all_label_path = os.path.join(self.dataset_path, 'Labels', 'AllLabels.csv')
      all_df = pd.read_csv(_all_label_path)
      del _all_label_path

      # Gather all Video IDs.
      vidlist = [video[:-4] for video in all_df.ClipID]

      # Validate Video ID.
      if name in vidlist:
         return True
      return False

   def _is_parsed(self, outdir, _warn = False):
      """Internal method and attribute to determine whether there is existing parsed data."""
      if not os.path.exists(outdir): # If the directory does not exist.
         raise OSError(f"The data path {outdir} does not exist.")
      if len(os.listdir(outdir)) == 0: # If the directory is empty.
         return False
      else: # Iterate over items in directory.
         for item in os.listdir(outdir):
            if item in ['train.tfrecords', 'validation.tfrecords', 'test.tfrecords']:
               if _warn:
                  raise FileExistsError(f"The file {item} already exists, if you want to overwrite data set 'overwrite' to True.")
               else:
                  self._parsed = True

   def _resize(self, image):
      """Utility method to resize an image to the necessary specifications."""
      return cv2.resize(image, dsize = (self.image_size[0], self.image_size[1]), interpolation = cv2.INTER_AREA)

   def _random_crop(self, image, crop_height, crop_width):
      """Utility method to randomly crop an image."""
      max_y = image.shape[0] - crop_height
      max_x = image.shape[1] - crop_width

      x = np.random.randint(0, max_x)
      y = np.random.randint(0, max_y)

      _crop_image = image[y: y + crop_height, x: x + crop_width]

      return self.detect_face(_crop_image)

   def _image_augment(self, image):
      """Utility method to apply augmentation techniques to an image."""
      if not isinstance(image, np.ndarray):
         raise TypeError(f"The image should be an np.ndarray, but you have provided a {type(image)}")

      # Mirror Flip
      _flipped = tf.image.flip_left_right(image).numpy()
      # Transpose Flip
      _transposed = tf.image.transpose(image).numpy()
      # Saturation
      _saturated = tf.image.adjust_saturation(image, 3).numpy()
      # Brightness
      _brightness = tf.image.adjust_brightness(image, 0.4).numpy()
      # Contrast
      _contrast = tf.image.adjust_contrast(image, lower = 0.0, upper = 1.0).numpy()

      # Resize and return list of augmented images.
      return [self._resize(_image) for _image in [_flipped, _transposed, _saturated, _brightness, _contrast]]

   def detect_face(self, image):
      """Proprietary method to detect a face from an image, using the class detector."""
      _face_detected = None
      image = self._resize(image)
      if self.detector_name == 'cascade': # Detect faces using the cascade classifier.
         try:
            faces = self.cascade.detectMultiScale(image, 1.3, 5)
            try:
               if len(faces) != 0:
                  x, y, w, h = faces[0]
                  _face_detected = image[y: y + h, x: x + w]
            except Exception as e:
               raise e
         except Exception as e:
            raise e
      elif self.detector_name == 'dnn': # Detect faces using the DNN detector.
         try:
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), swapRB = False, crop = False)
            self.detector.setInput(blob)
            faces = self.detector.forward()
            try:
               if faces.shape[2] != 0:
                  _c = faces[0, 0, 0, 2]
                  if _c < 0.5:
                     pass
                  _coords = faces[0, 0, 0, 3:7] * \
                            np.array([self.image_size[0], self.image_size[1],
                                      self.image_size[0], self.image_size[1]])
                  (x, y, xe, ye) = _coords.astype("int")
                  _face_detected = image[y: ye, x: xe]
            except Exception as e:
               raise e
         except Exception as e:
            raise e
      else:
         raise AttributeError("You are using an invalid detector for face detection.")

      if len(_face_detected) > 0:
         return self._resize(_face_detected)

   @staticmethod
   def get_picture_label(image, label_df):
      """Proprietary method to get the label associate with a picture."""
      _encountered_error = False
      _video = image.split("/")[-2]
      label_series = label_df.loc[((label_df.ClipID == _video + '.avi') | (label_df.ClipID == _video + '.mp4'))]
      try:
         indx = label_series.values[0]
         label = np.array([label_series.Boredom.get(indx)],
                          [label_series.Engagement.get(indx)],
                          [label_series.Confusion.get(indx)],
                          [label_series.Frustration.get(indx)])
         _one_hot = label.astype(np.uint8) # Turn into one-hot-encoded vectors.
      except:
         #print(f"Encountered error in image {image}.")
         _one_hot = ''
         _encountered_error = True
      finally:
         del _video
      return _one_hot, _encountered_error

   @staticmethod
   def _bytes_feature(value):
      """Returns a list of bytes from a string/byte."""
      if isinstance(value, type(tf.constant(0))):
         value = value.numpy()
      return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

   def _write_tf_record(self, outdir, data_augmentation = False):
      """Proprietary method to write and store TFRecord of dataset."""
      if not os.path.exists(outdir): # Use or create output dataset.
         try:
            os.makedirs(outdir)
         except OSError as ose:
            raise ose
         except Exception as e:
            raise e

      # Create objects to iterate over.
      _dataset_path = os.path.join(self.dataset_path, 'Dataset')
      _iter_objs = [('train', os.path.join(_dataset_path, 'Train'), self.train_labels),
                    ('validation', os.path.join(_dataset_path, 'Validation'), self.validation_labels),
                    ('test', os.path.join(_dataset_path, 'Test'), self.test_labels)]

      # Write TFRecord.
      for name, dataset, label_df in tqdm(_iter_objs):
         # Create writer and get images from set.
         writer = tf.io.TFRecordWriter(os.path.join(outdir, f'{name}.tfrecords'))
         _image_set_path = self._get_images(dataset)
         for image_path in tqdm(_image_set_path, total = len(_image_set_path)):
            # Get image and detect face.
            _image = cv2.imread(image_path)[..., ::-1]
            _face = self.detect_face(_image)

            # Read label.
            label, _encountered_error = self.get_picture_label(image_path, label_df)
            if _encountered_error:
               continue

            # Create Feature.
            if data_augmentation:
               _images = self._image_augment(_image)
            else:
               _images = _image
            try:
               for image in _images:
                  feature = {
                     'label': self._bytes_feature(tf.compat.as_bytes(label.tostring())),
                     'image': self._bytes_feature(tf.compat.as_bytes(image.tostring()))
                  }
                  # Create example protocol buffer.
                  example = tf.train.Example(features = tf.train.Features(feature = feature))

                  # Serialize to string and write to file.
                  writer.write(example.SerializeToString())
            except Exception as e:
               raise e
            finally:
               del _image, _images, _face, label, _encountered_error
         writer.close()

   def decode(self, serialized):
      """Proprietary method to parse an image and label from a given example."""
      _shape = (self.image_size[0], self.image_size[1], 3)

      # Define and construct parser.
      _features = tf.io.parse_single_example(
         serialized,
         features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string)
         }
      )

      # Convert, cast, and reshape data.
      try:
         _image = tf.io.decode_raw(_features['image'], tf.uint8)
         _label = tf.io.decode_raw(_features['label'], tf.uint8)
         image = tf.convert_to_tensor(tf.reshape(_image, _shape))
         label = tf.cast(_label, tf.float32)
      except Exception as e:
         raise e
      finally:
         del _features

      return image, label

   def construct(self, directory, overwrite = False, data_augmentation = False):
      """Primary method (called by user) to construct TFRecords from the dataset."""
      if overwrite is False:
         self._is_parsed(directory, _warn = True)
      self._write_tf_record(directory, data_augmentation = data_augmentation)

   def load(self, path = None):
      """Primary method (called by user) to gather the TFRecord files of the dataset and convert to usable format."""
      if path:
         if not os.path.exists(path):
            raise OSError(f"The directory path {path} does not exist.")
      else:
         path = os.path.join(os.path.dirname(__file__), 'data/tfrecords')

      # Load training data.
      _training_path = os.path.join(path, 'train.tfrecords')
      if not os.path.exists(_training_path):
         raise FileNotFoundError(f"The file {_training_path} does not exist, please re-construct dataset.")
      try:
         _train_set = tf.data.TFRecordDataset(_training_path)
         _train_set = _train_set.map(self.decode)
         _train_set = _train_set.shuffle(1)
         _train_set.batch(32)
         self.train_data = _train_set
      except Exception as e:
         raise e
      finally:
         del _training_path, _train_set

      # Load validation data.
      _validation_path = os.path.join(path, 'validation.tfrecords')
      if not os.path.exists(_validation_path):
         raise FileNotFoundError(f"The file {_validation_path} does not exist, please re-construct dataset.")
      try:
         _validation_set = tf.data.TFRecordDataset(_validation_path)
         _validation_set = _validation_set.map(self.decode)
         _validation_set = _validation_set.shuffle(1)
         _validation_set.batch(32)
         self.validation_data = _validation_set
      except Exception as e:
         raise e
      finally:
         del _validation_path, _validation_set

      # Load test data.
      _test_path = os.path.join(path, 'test.tfrecords')
      if not os.path.exists(_test_path):
         raise FileNotFoundError(f"The file {_test_path} does not exist, please re-construct dataset.")
      try:
         _test_set = tf.data.TFRecordDataset(_test_path)
         _test_set = _test_set.map(self.decode)
         _test_set = _test_set.shuffle(1)
         _test_set.batch(32)
         self.test_data = _test_set
      except Exception as e:
         raise e
      finally:
         del _test_path, _test_set

      return self.train_data, self.validation_data, self.test_data

if __name__ == '__main__':
   # Parse command line arguments.
   ap = argparse.ArgumentParser()
   ap.add_argument('-e', '--detector', default = 'dnn',
                   help = "The detector to use for facial detection.")
   ap.add_argument('-d', '--directory', default = os.path.join(os.path.dirname(__file__), 'data/tfrecords/'),
                   help = "The directory to save the TFRecords to.")
   ap.add_argument('-o', '--overwrite', default = False, type = bool,
                   help = "If you want to overwrite existing parsed data.")
   args = vars(ap.parse_args())

   # Perform preprocessing.
   Dataset(detector = args['detector']).construct(
      args['directory'],
      overwrite = args['overwrite'],
      data_augmentation = True
   )
else:
   print(__name__)


