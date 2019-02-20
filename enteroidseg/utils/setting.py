import sys
import yaml

with open('config/settings.yaml') as f:
  settings = yaml.safe_load(f.read())

with open('config/seg_params.yaml') as f:
  seg_params = yaml.safe_load(f.read())

paths = settings['paths']

max_px_val = settings['max_px_val']
