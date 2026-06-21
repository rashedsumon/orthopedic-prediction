import os
import glob
import pandas as pd
import kagglehub

def load_orthopedic_data():
    """
    Downloads the dataset via kagglehub and returns it as a pandas DataFrame.
    Uses the 3-class version (Normal, Disk Hernia, Spondylolisthesis).
    """
    # Download latest version of the dataset
    path = kagglehub.dataset_download("uciml/biomechanical-features-of-orthopedic-patients")
    
    # Find the 3-class CSV file in the downloaded path
    csv_files = glob.glob(os.path.join(path, "*column_3C*.csv"))
    
    if not csv_files:
        raise FileNotFoundError("Could not find the 3-class CSV file in the downloaded dataset.")
        
    # Load and return the dataframe
    df = pd.read_csv(csv_files[0])
    return df

if __name__ == "__main__":
    # Quick test to make sure downloader works
    try:
        data = load_orthopedic_data()
        print("Dataset successfully loaded! Shape:", data.shape)
        print(data.head(2))
    except Exception as e:
        print("Error loading data:", e)