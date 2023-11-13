import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os
import glob

# Parameters
source_dir = 'npy_img/'  # Directory with original files
save_to_dir = 'npy_img_sim'  # Directory to save modified files

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

     # Visualize the data
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(wo_tumor_input, cmap='gray')
    axs[0, 0].set_title('Input without Tumor')
    axs[0, 1].imshow(wo_tumor_target, cmap='gray')
    axs[0, 1].set_title('Target without Tumor')

    # Assuming you have corresponding 'with tumor' data to display
    # Replace 'with_tumor_input' and 'with_tumor_target' with your actual data variables
    axs[1, 0].imshow(with_tumor_input, cmap='gray')
    axs[1, 0].set_title('Input with Tumor')
    axs[1, 1].imshow(with_tumor_target, cmap='gray')
    axs[1, 1].set_title('Target with Tumor')
    axs[0, 0].axis('off')
    axs[1, 0].axis('off')
    axs[0, 1].axis('off')
    axs[1, 1].axis('off')
    #save to npy_img_comparion folder
    plt.savefig(f'npy_img_comparison/{base_name}.png')
    plt.show()
    print(f"Processed {base_name}")
