import numpy as np
from skimage.transform import radon, iradon
import os
import matplotlib.pyplot as plt

def process_images(path):

    #Ladda bilderna
    if path.endswith('.npy'):
        images = [np.load(path)]
    else:
        images = [np.load(os.path.join(path, file)) for file in os.listdir(path) if file.endswith('.npy')]

    # Konstanten för luftens µ-värde
    mu_air = 0

    processed_images = []

    for image in images:
        
        # Beräkna radontransformen
        theta = np.linspace(0., 180., max(image.shape), endpoint=False)
        sinogram = radon(image, theta=theta, circle=True)

        
        # Skala om för att få linjeintegralen dimensionslös
        pixel_size = 0.2645833333  
        sinogram *= pixel_size


        N0 = 1 # Hur bestämmer vi detta?

        # Osäker om detta är rätt
        lambda_values = N0 * np.exp(-sinogram)
        N = np.random.poisson(lambda_values)
        p = -np.log(N / N0)

        # Invertera radontransformen
        reconstruction = iradon(p, theta=theta, circle=True)

        # Skala om tillbaka till pixlar
        mu_water = 0.2 # µ-värde för vatten
        reconstruction *= pixel_size
        CT = 1000 * (reconstruction - mu_water) / (mu_water - mu_air)

        # Visa bilderna
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap=plt.cm.Greys_r)
        plt.title('Originalbild')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(sinogram, cmap=plt.cm.Greys_r)
        plt.title('Ursprungligt Sinogram')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(CT, cmap=plt.cm.Greys_r)
        plt.title('Rekonstruerad Bild')
        plt.axis('off')

        plt.show()

    return processed_images

path = "./npy_img/L506_0_input.npy"  # Ersätt med faktisk sökväg
processed_images = process_images(path)

