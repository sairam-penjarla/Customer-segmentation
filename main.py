#Importing the Libraries
import numpy as np
import pandas as pd
import warnings
import sys
from src.preprocessing import PreprocessingSteps
from src.model_training import ClusteringModel

preprocessor = PreprocessingSteps()
model = ClusteringModel()


data = pd.read_csv("dataset/marketing_campaign.csv", sep="\t").dropna()

# preprocessing
PCA_data = preprocessor.run(data)
# Training
model.fit(PCA_data)
# Predictions
model.predict(PCA_data)