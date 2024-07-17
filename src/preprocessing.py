import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class FeatureEngineering:
    def run(self, data):
        # Feature Engineering
        # Calculate age of customers based on birth year
        data["Age"] = 2021 - data["Year_Birth"]

        # Calculate total spending across various items
        data["Spent"] = data[["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]].sum(axis=1)

        # Determine living situation based on marital status
        living_situation_map = {
            "Married": "Partner",
            "Together": "Partner",
            "Absurd": "Alone",
            "Widow": "Alone",
            "YOLO": "Alone",
            "Divorced": "Alone",
            "Single": "Alone"
        }
        data["Living_With"] = data["Marital_Status"].replace(living_situation_map)

        # Calculate total number of children in the household
        data["Children"] = data["Kidhome"] + data["Teenhome"]

        # Calculate family size based on living situation and children count
        data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner": 2}) + data["Children"]

        # Create a binary indicator for parenthood
        data["Is_Parent"] = (data["Children"] > 0).astype(int)

        # Segment education levels into three groups
        education_map = {
            "Basic": "Undergraduate",
            "2n Cycle": "Undergraduate",
            "Graduation": "Graduate",
            "Master": "Postgraduate",
            "PhD": "Postgraduate"
        }
        data["Education"] = data["Education"].replace(education_map)

        # Rename spending columns for clarity
        data = data.rename(columns={
            "MntWines": "Wines",
            "MntFruits": "Fruits",
            "MntMeatProducts": "Meat",
            "MntFishProducts": "Fish",
            "MntSweetProducts": "Sweets",
            "MntGoldProds": "Gold"
        })

        # Drop redundant features
        features_to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
        data = data.drop(features_to_drop, axis=1)
        return data



class PreprocessingSteps:
    def __init__(self) -> None:
        self.fe = FeatureEngineering()

    def plot_data_3D(self, PCA_data):
        # Extract PCA components
        x = PCA_data["col1"]
        y = PCA_data["col2"]
        z = PCA_data["col3"]
        
        # Plotting in 3D
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, marker="x")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
        plt.show()
        plt.savefig("./plots/Raw data.png")

    def run(self, data):
        data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], format="%d-%m-%Y")
        d1 = data['Dt_Customer'].max()
        data['Customer_For'] = (d1 - data['Dt_Customer']).dt.days
        data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce")

        data = self.fe.run(data)

        # Filter out outliers based on age and income
        data = data[(data["Age"] < 90) & (data["Income"] < 600000)]
        # Get list of categorical variables
        object_cols = data.select_dtypes(include=['object']).columns.tolist()

        # Label encode the categorical variables
        LE = LabelEncoder()
        data[object_cols] = data[object_cols].apply(lambda col: LE.fit_transform(col))
        
        # Drop columns directly from the original data variable
        cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response']
        data.drop(cols_del, axis=1, inplace=True)

        # Scale the remaining features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns)
        
        # Principle Component Analysis
        pca = PCA(n_components=3).fit(scaled_data)
        PCA_data = pd.DataFrame(pca.transform(scaled_data), columns=(["col1","col2", "col3"]))
        
        self.plot_data_3D(PCA_data)

        return PCA_data