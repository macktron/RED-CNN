import numpy as np
from skimage.transform import radon, iradon
import os
import matplotlib.pyplot as plt
import glob 


def process_images(path, N0 = 10**10, mu_water=0.2,mu_air=0, plot=False, save=False, save_dir = "npy_img_physics_simulation"):
    
    if path.endswith('.npy'):
        print('Path is a file, load directory instead')
        exit()

    if not os.path.exists(save_to_dir):
        print('Directory does not exist')
        exit()

    if not os.path.exists(source_dir):
        print('Directory does not exist')
        exit()

    images = [np.load(os.path.join(path, file)) for file in os.listdir(path) if file.endswith('.npy')]
    processed_images = []

    for image in images:
        # Beräkna radontransformen
        theta = np.linspace(0., 180., max(image.shape), endpoint=False)
        sinogram = radon(image, theta=theta, circle=True)

        # Skala om för att få linjeintegralen dimensionslös
        pixel_size = 0.2645833333
        sinogram *= pixel_size

        # Beräkna antalet fotoner som har passerat genom objektet
        lambda_values = N0 * np.exp(-sinogram)
        N = np.random.poisson(lambda_values)

        p = -np.log(N / N0 + 1/10**11 )  # Lägg till  liten konstant för att undvika log med noll
        
        
        # Invertera radontransformen
        reconstruction = iradon(p, theta=theta, circle=True)

        # Skala om
        reconstruction *= pixel_size
        CT = 1000 * (reconstruction - mu_water) / (mu_water - mu_air)

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))  # Adjusted to 1 row, 2 columns

            # Original Image
            ax = axes[0]
            ax.imshow(image, cmap=plt.cm.Greys_r)
            ax.set_title('Original Image')
            ax.axis('off')

            # Reconstructed CT Image
            ax = axes[1]
            ax.imshow(CT, cmap=plt.cm.Greys_r)
            ax.set_title('Reconstructed CT Image')
            ax.axis('off')

            plt.show()

        if save: 
            for i, input_file_path in enumerate(glob.glob(os.path.join(path, '*input.npy'))):
                input_file_base_name = os.path.basename(input_file_path)
                base_name = input_file_base_name.replace('_input.npy', '')
                target_file_base_name = base_name + '_target.npy'
                target_file_path = os.path.join(path, target_file_base_name)

                result_input_path = os.path.join(save_to_dir, input_file_base_name)
                result_target_path = os.path.join(save_to_dir, target_file_base_name)

                np.save(result_input_path, processed_images[i])




# Parameters
source_dir = 'npy_img/'  # Directory with original files
save_to_dir = 'npy_img_physics_simulation'  # Directory to save modified files
N0 = 10**12 # Antal fotoner som träffar detektorerna i tomma luften


processed_images = process_images(source_dir, plot=True, N0=N0, mu_water=0.2, mu_air=0,save=False, save_dir=save_to_dir)


