import os
import nibabel as nib
from multiprocessing import Pool, cpu_count

def verify_nifti_file(file_path):
    try:
        img = nib.load(file_path)
        img_data = img.get_fdata()
        return file_path, True, None
    except Exception as e:
        return file_path, False, str(e)

def verify_all_nifti_files(directory, num_workers=None):
    nifti_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.nii') or f.endswith('.nii.gz')]
    
    if num_workers is None:
        num_workers = cpu_count()
        
    with Pool(num_workers) as pool:
        results = pool.map(verify_nifti_file, nifti_files)
    
    corrupted_files = []
    
    for result in results:
        file_path, is_valid, error = result
        if is_valid:
            pass
            print(f"{file_path}: File is valid")
        else:
            pass
            print(f"{file_path}: File is corrupted or invalid ({error})")
            corrupted_files.append(file_path)
    
    return corrupted_files

if __name__ == '__main__':
    dataset = 'niftiShort'
    data_path = '../datasets/nifti/'
    data_dir = f'{data_path}{dataset}'
    # Example usage
    d = "/Users/askelundsgaard/Documents/datalogi/6-semester/Bachelor/XAI-in-Medical-Image-Analysis/datasets/nifti/niftiShort"
    corrupted_files = verify_all_nifti_files(d)

    if corrupted_files:
        print("\nList of corrupted files:")
        for file in corrupted_files:
            print(file)
    else:
        print("\nNo corrupted files found.")

