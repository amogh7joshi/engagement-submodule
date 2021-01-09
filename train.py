#!/usr/bin/env python3
# -*- coding = utf-8
import os
import sys

import numpy as np

from tqdm import tqdm

import tensorflow as tf

from dataset import Dataset
from model import light_network
from util import create_train_step_dirs

# Load dataset.
data = Dataset()
data.load()

# Training/Testing Functions.
@tf.function
def train_step(x, y):
   with tf.GradientTape() as tape:
      logits = model(x)
      loss_value = tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = logits)
   grads = tape.gradient(loss_value, model.trainable_weights)
   optimizer.apply_gradients(zip(grads, model.trainable_weights))
   train_loss_avg.update_state(loss_value)
   train_accuracy.update_state(y, logits)
   return loss_value

@tf.function
def test_step(x, y, set_name):
   logits = model(x)
   if set_name == 'validation':
      validation_accuracy.update_state(y, logits)
   else:
      test_accuracy.update_state(y, logits)

# Define Model Architectures.
model = light_network()
optimizer = tf.keras.optimizers.Adam(lr = 0.01)
train_loss_avg = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.MeanSquaredError()
validation_accuracy = tf.keras.metrics.MeanSquaredError()
test_accuracy = tf.keras.metrics.MeanSquaredError()

# Create and Validate Training Logs and Model Checkpoint Directories.
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
checkpoint_dir = os.path.join(os.path.dirname(__file__), 'models')
create_train_step_dirs(log_dir, checkpoint_dir)
train_summary_writer = tf.summary.create_file_writer(log_dir)

# Train Model.
for epoch in range(10):
   # Training batch.
   for x_batch, y_batch in tqdm(data.train_data, total = 1517):
      loss_value = train_step(x_batch, y_batch)

   # Validation batch.
   for x_val_batch, y_val_batch in data.validation_data:
      test_step(x_val_batch, y_val_batch)

   # Training and validation accuracy/loss.
   train_acc = train_accuracy.result()
   train_accuracy.reset_states()
   validation_acc = validation_accuracy.result()
   validation_accuracy.reset_states()

   # Write training log.
   with train_summary_writer.as_default():
      tf.summary.scalar('Training Loss', train_loss_avg.result(), step = epoch)
      tf.summary.scalar('Training MSE', train_acc, step = epoch)
      tf.summary.scalar('Validation MSE', validation_acc, step = epoch)

   # Save every fifth model.
   if epoch % 5 == 0:
      tf.keras.models.save_model(model, os.path.join(checkpoint_dir, f'Epoch-{model}.h5'), save_format = 'h5')


