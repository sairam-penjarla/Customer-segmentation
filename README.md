# Project Title: Customer Segmentation Analysis

##### Overview:
- This project aims to segment customers based on their demographic and purchasing behavior using clustering techniques.

- dataset link: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis


##### Files:
- `preprocessing.py`: Contains classes for data preprocessing steps.
- `model_training.py`: Contains classes for clustering model training and visualization.
- `main.py`: Main script to run preprocessing, model training, and prediction.

##### Libraries Used:
- `numpy`
- `pandas`
- `scikit-learn`
- `yellowbrick`
- `matplotlib`

##### Installation:
1. Clone the repository:
   ```
   git clone https://github.com/sairam-penjarla/Customer-segmentation.git
   cd Customer-segmentation
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

##### Usage:
- Ensure your dataset (`marketing_campaign.csv`) is placed in the `dataset` folder.
- Run the `main.py` script to execute the entire pipeline:
  ```
  python main.py
  ```

##### Outputs:
- The script generates 3D plots of the clustered data.