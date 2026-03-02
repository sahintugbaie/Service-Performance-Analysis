import pandas as pd
import os

print("Testing Excel file loading...")
excel_path = "attached_assets/veriler.xlsx"

if os.path.exists(excel_path):
    print(f"File exists at {excel_path}")
    print(f"File size: {os.path.getsize(excel_path)} bytes")
    
    try:
        # Try loading with openpyxl engine
        print("Trying to load with openpyxl engine...")
        df = pd.read_excel(excel_path, engine="openpyxl")
        print("Success with openpyxl!")
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First few rows:")
        print(df.head())
    except Exception as e:
        print(f"Failed with openpyxl: {str(e)}")
        
    try:
        # Try loading with xlrd engine
        print("\nTrying to load with xlrd engine...")
        df = pd.read_excel(excel_path, engine="xlrd")
        print("Success with xlrd!")
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First few rows:")
        print(df.head())
    except Exception as e:
        print(f"Failed with xlrd: {str(e)}")
        
    try:
        # Try loading with odf engine
        print("\nTrying to load with odf engine...")
        df = pd.read_excel(excel_path, engine="odf")
        print("Success with odf!")
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First few rows:")
        print(df.head())
    except Exception as e:
        print(f"Failed with odf: {str(e)}")
        
else:
    print(f"File does not exist at {excel_path}")