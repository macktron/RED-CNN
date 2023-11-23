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
    N = np.maximum(N, 1)  # Filter out values that are too low

    # Beräkna linjeintegralen
    p = -np.log(N / N0)

    # Invertera radontransformen
    reconstruction = iradon(p, theta=theta, circle=False)

    # Skala om
    reconstruction *= pixel_size_cm
    return reconstruction
    simulated_images = 1000 * (reconstruction - mu_water) / (mu_water - mu_air)

    # Rescale to 0,1
    simulated_images -= np.min(simulated_images)
    simulated_images /= np.max(simulated_images)
    simulated_images *= np.max(image)
    return simulated_images

def plot_images(original_image, CT_image, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax = axes[0]
    ax.imshow(original_image, cmap=plt.cm.Greys_r)
    ax.set_title('Original Image')
    ax.axis('off')
    ax = axes[1]
    ax.imshow(CT_image, cmap=plt.cm.Greys_r)
    ax.set_title('Reconstructed CT Image')
    ax.axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def save_images(CT_image, save_dir, image_name):
    np.save(os.path.join(save_dir, image_name), CT_image)
    plt.imsave(os.path.join(save_dir, image_name.replace(".npy", ".pdf")), CT_image, cmap=plt.cm.Greys_r)

def get_noise(original_image, CT_image):

    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(original_image, theta=theta, circle=False)
    original_reconstructed = iradon(sinogram, theta=theta, circle=False)
    variance_original = np.std(original_reconstructed)**2
    variance_CT = np.std(CT_image)**2

    print(f"Original max: {np.max(original_image)} min: {np.min(original_image)}")
    print(f"CT max: {np.max(CT_image)} min: {np.min(CT_image)}")

    print(f"Dose: {100*variance_original / variance_CT}%")


source_dir = 'npy_img/'  # Directory with original files

N0 = 10**5
mu_water = 0.2
mu_air = 0
save = False
plot = True

save_dir = "npy_img_physics_simulation"
save_dir_pdf = "pdf_img_physics_simulation"

image_names = get_image_names(source_dir)

for image_name in image_names:
    
    image_path = os.path.join(source_dir, image_name)
    image = np.load(image_path)

    CT = process_image(image_path, N0, mu_water, mu_air)
    get_noise(image, CT)

    if save:
        save_images(CT, save_dir, image_name)
    if plot:
        plot_images(np.load(image_path), CT, save_path=os.path.join(save_dir_pdf, image_name.replace(".npy", ".pdf")) if save else None)