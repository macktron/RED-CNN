import numpy as np
import os
from skimage.transform import radon, iradon
import matplotlib.pyplot as plt

def get_image_names(path):
    if path.endswith('.npy'):
        return [path]
    if not os.path.exists(path):
        raise ValueError('Provided directory or file does not exist.')
    return [file for file in os.listdir(path) if file.endswith('target.npy')]

def save_images(CT_image, save_dir, image_name):
    input_name = image_name.replace("target", "input")
    np.save(os.path.join(save_dir, input_name), CT_image)
    #plt.imsave(os.path.join(save_dir, image_name.replace(".npy", ".pdf")), CT_image, cmap=plt.cm.Greys_r)

def plot_images(image_name, CT_image,save=False, save_path=None, plot=True, N0 = 10**8):
    original_image = np.load(image_name)
    basename = os.path.basename(image_name)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax = axes[0]
    ax.imshow(original_image, cmap=plt.cm.Greys_r)
    ax.set_title('Original Image')
    ax.axis('off')
    ax = axes[1]
    ax.imshow(CT_image, cmap=plt.cm.Greys_r)
    ax.set_title(f'Reconstructed CT Image N0={N0: .2e}')
    ax.axis('off')
    if save and save_path:
        name = basename.replace(".npy", ".pdf")
        path = os.path.join(save_path, name)
        print(f"Saving to {path}")
        plt.savefig(path)
    if plot:
        plt.show()


def process_image(image_path, N0 = 10**8, mu_water = 0.2, mu_air = 0.0, pixel_size_cm = 0.0264583333):

    # Load image
    image = np.load(image_path)

    # Beräkna radontransformen
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta, circle=False)

    # Skala om för att få linjeintegralen dimensionslös
    sinogram *= pixel_size_cm

    # Beräkna antalet fotoner som har passerat genom objektet
    lambda_values = N0 * np.exp(-sinogram)

    # Simulera poissonfördelad slump
    N = np.random.poisson(lambda_values)
    #N = np.maximum(N, 1)  # Filter out values that are too low

    # Beräkna linjeintegralen
    p = -np.log(N / N0)

    # Invertera radontransformen
    reconstruction = iradon(p, theta=theta, circle=False)

    # Skala om
    reconstruction /= pixel_size_cm
    
    return reconstruction


source_dir = 'npy_img/'  # Directory with original files
save_dir_npy = "lowdose_simulation/npy_img_physics_simulation/"
save_dir_pdf = "lowdose_simulation/pdf_img_physics_simulation/"


N0 = 5*10**3
mu_water = 0.2
mu_air = 0
save = True
plot = False


image_names = get_image_names(source_dir)

for image_name in image_names:
    
    image_path = os.path.join(source_dir, image_name)
    image = np.load(image_path)

    print(f"Processing {image_path}")
    CT = process_image(image_path, N0, mu_water, mu_air)

    if save:
        print(f"Saving to {save_dir_npy}")
        save_images(CT, save_dir_npy, image_name)
    if plot or save:
        plot_images(image_path, CT, plot=plot, save=save, N0=N0, save_path= save_dir_pdf)