#!/usr/bin/env python3
import os

def create_train_step_dirs(log_dir, model_dir):
   """Create directories which contain model training logs and model checkpoints."""
   if not os.path.exists(log_dir):
      os.makedirs(log_dir)
   if not os.path.exists(model_dir):
      os.makedirs(model_dir)


