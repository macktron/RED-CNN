import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os
import glob

# Parameters
source_dir = 'npy_img/'  # Directory with original files
save_to_dir = 'npy_img_sim'  # Directory to save modified files
comparison_dir = 'npy_img_comparison'  # Directory to save comparison images

if not os.path.exists(save_to_dir):
    print('Directory does not exist')
    exit()

if not os.path.exists(source_dir):
    print('Directory does not exist')
    exit()

# Processing all files ending with 'input.npy' and modifying the corresponding 'target.npy' files aswell
for input_file_path in glob.glob(os.path.join(source_dir, '*input.npy')):
    
    input_file_base_name = os.path.basename(input_file_path)
    base_name = input_file_base_name.replace('_input.npy', '')
    target_file_base_name = base_name + '_target.npy'
    target_file_path = os.path.join(source_dir, target_file_base_name)  # Assuming target files are in the same directory as input files

    result_input_path = os.path.join(save_to_dir, input_file_base_name)
    result_target_path = os.path.join(save_to_dir, target_file_base_name)

    # Load the data
    wo_tumor_input = np.load(input_file_path)
    wo_tumor_target = np.load(target_file_path)
    with_tumor_input = np.load(result_input_path)
    with_tumor_target = np.load(result_target_path)

    # Visualize the data with increased figure size and DPI
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=300)
    axs[0, 0].imshow(wo_tumor_input, cmap='gray', interpolation='none')
    axs[0, 0].set_title('Without tumor input')
    axs[0, 1].imshow(wo_tumor_target, cmap='gray', interpolation='none')
    axs[0, 1].set_title('Without tumor target')
    axs[1, 0].imshow(with_tumor_input, cmap='gray', interpolation='none')
    axs[1, 0].set_title('With tumor input')
    axs[1, 1].imshow(with_tumor_target, cmap='gray', interpolation='none')
    axs[1, 1].set_title('With tumor target')

    for ax in axs.flat:
        ax.axis('off')

    plt.savefig(os.path.join(comparison_dir, f'{base_name}.png'))
    print(f"Processed {base_name}")