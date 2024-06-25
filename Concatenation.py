import os
import pandas as pd

def concatenate_csv_files(folder_path, output_folder):
    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Sort files to ensure consistent concatenation order
    csv_files.sort()

    # Initialize an empty DataFrame for the result
    concatenated_df = pd.DataFrame()

    # Track if 'label' column has been added
    label_added = False

    for i, file in enumerate(csv_files):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        
        # Drop any columns named 'Unnamed'
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Prefix columns with file index (except 'label')
        prefixed_columns = {col: f"{col}_{i+1}" for col in df.columns if col.lower() != 'label'}
        df = df.rename(columns=prefixed_columns)
        
        # Rename 'label' column if it exists
        if 'label' in df.columns:
            df = df.rename(columns={'label': 'label'})
        
        # Append the 'label' column as the last column if it exists
        if 'label' in df.columns and not label_added:
            concatenated_df['label'] = df['label']
            label_added = True
        
        # Concatenate the DataFrames
        concatenated_df = pd.concat([concatenated_df, df.drop(columns=['label'], errors='ignore')], axis=1)
        
        # Save the intermediate result after each concatenation
        intermediate_output_path = os.path.join(output_folder, f"concatenated_up_to_{i+1}.csv")
        
        # Reorder columns with 'label' as the last column
        if 'label' in concatenated_df.columns:
            columns_ordered = [col for col in concatenated_df.columns if col != 'label'] + ['label']
            concatenated_df = concatenated_df[columns_ordered]
        
        concatenated_df.to_csv(intermediate_output_path, index=False)
        print(f"Saved {intermediate_output_path}")

    # Final concatenated file
    final_output_path = os.path.join(output_folder, "final_concatenated.csv")
    
    # Reorder columns with 'label' as the last column
    if 'label' in concatenated_df.columns:
        columns_ordered = [col for col in concatenated_df.columns if col != 'label'] + ['label']
        concatenated_df = concatenated_df[columns_ordered]
    
    concatenated_df.to_csv(final_output_path, index=False)
    print(f"Final concatenated file saved as {final_output_path}")

# Usage
folder_path = 'Features'
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)
concatenate_csv_files(folder_path,output_folder)
