#!/bin/bash
sudo apt update && sudo apt install unzip
echo "Downloading dataset..."
curl -L -o ./ecg-data-for-sleep-apnea-detection.zip\
  https://www.kaggle.com/api/v1/datasets/download/ucimachinelearning/ecg-data-for-sleep-apnea-detection
unzip ecg-data-for-sleep-apnea-detection.zip -d .
rm ecg-data-for-sleep-apnea-detection.zip
