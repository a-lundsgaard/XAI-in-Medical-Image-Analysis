import os
import nibabel as nib
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def verify_nifti_file(file_path):
    try:
        img = nib.load(file_path)
        img_data = img.get_fdata()
        return file_path, True, None
    except Exception as e:
        return file_path, False, str(e)

# Move verify_with_progress outside to make it pickleable
def verify_with_progress(args):
    file_path, pbar_pos, total_files = args
    result = verify_nifti_file(file_path)
    with tqdm(total=total_files, position=pbar_pos, leave=False) as pbar:
        pbar.update(1)
    return result

def verify_all_nifti_files(directory, num_workers=None):
    nifti_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.nii') or f.endswith('.nii.gz')]
    
    if num_workers is None:
        num_workers = cpu_count()
    
    corrupted_files = []

    # Create a list of arguments for verify_with_progress
    tasks = [(file_path, idx, len(nifti_files)) for idx, file_path in enumerate(nifti_files)]

    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(verify_with_progress, tasks), total=len(nifti_files), desc="Verifying NIfTI files"))
    
    for result in results:
        file_path, is_valid, error = result
        if is_valid:
            # print(f"{file_path}: File is valid")
            pass
        else:
            # print(f"{file_path}: File is corrupted or invalid ({error})")
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
