# ASL gesture trainer

A Flask-based web application for **learning and practicing** the American Sign Language (ASL) alphabet through real-time hand gesture recognition

## Features

- **Practice mode**:  
  Show gestures to your camera - the app recognizes letters in real time and gives feedback
- **Learning mode**:  
  You can see all ASL alphabet letters and view reference images for each gesture

## Dataset

This project uses the **[ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)** dataset from Kaggle for training the model  

## How to run

```bash
git clone https://github.com/sofa797/asl_gesture_trainer.git
cd asl_gesture_trainer
pip install -r requirements.txt
python app.py
