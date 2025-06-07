import os
import zipfile

def zip_subfolders(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for folder_name in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder_name)
        
        if os.path.isdir(folder_path):
            zip_path = os.path.join(output_dir, f"{folder_name}.zip")
            print(f"Zipping {folder_name} → {zip_path}")
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        abs_file = os.path.join(root, file)
                        rel_path = os.path.relpath(abs_file, start=folder_path)
                        zipf.write(abs_file, arcname=rel_path)

    print("Zipping completed.")

# Example usage:
input_directory = r"D:\ACADEMICS\THESIS\Datasets\WDF\WildDeepfake\wdf_final_real"
output_directory = r"D:\ACADEMICS\THESIS\Datasets\WDF\WildDeepfake\wdf_zipped_real"

zip_subfolders(input_directory, output_directory)
