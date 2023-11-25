import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os
import glob

def generate_2d_gaussian_noise(image_shape, center, sigma=10, max_amplitude=1, percentage=0.5):
    x_center, y_center = center
    x_center *= image_shape[1]  # Convert relative position to absolute
    y_center *= image_shape[0]

    x = np.arange(0, image_shape[1], 1)
    y = np.arange(0, image_shape[0], 1)
    x, y = np.meshgrid(x, y)

    amplitude = max_amplitude * percentage
    gaussian = amplitude * np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * sigma ** 2))
    gaussian[gaussian < amplitude * np.exp(-3**2 / 2)] = 0
    return gaussian

class ImageEditor:
    def __init__(self, input_file_path, target_file_path, output_file_path, output_target_file_path, sigma, percentage=0.5):
        self.original_image = np.load(input_file_path)
        self.original_target_image = np.load(target_file_path)
        self.max_amplitude = np.max(self.original_image)
        self.percentage = percentage
        self.current_image = np.copy(self.original_image)
        self.current_target_image = np.copy(self.original_target_image)
        self.history = []  # List to store history of image states
        self.output_file_path = output_file_path
        self.output_target_file_path = output_target_file_path
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.current_image, cmap='gray')
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.sigma = sigma

        # Save button
        self.ax_save = plt.axes([0.8, 0.05, 0.1, 0.075])
        self.btn_save = Button(self.ax_save, 'Save')
        self.btn_save.on_clicked(self.save_image)

        # Undo button
        self.ax_undo = plt.axes([0.65, 0.05, 0.1, 0.075])
        self.btn_undo = Button(self.ax_undo, 'Undo')
        self.btn_undo.on_clicked(self.undo_last_change)

    def onclick(self, event):
        if event.xdata is not None and event.ydata is not None and event.inaxes == self.ax:
            self.store_history()

            center = (event.xdata / self.current_image.shape[1], event.ydata / self.current_image.shape[0])
            gaussian_noise = generate_2d_gaussian_noise(self.current_image.shape, center, sigma = self.sigma, max_amplitude=self.max_amplitude, percentage=self.percentage)
            self.apply_gaussian_noise(gaussian_noise)
            self.update_image()

    def apply_gaussian_noise(self, gaussian_noise):
        self.current_image = np.clip(self.current_image + gaussian_noise, 0, 255)
        self.current_target_image = np.clip(self.current_target_image + gaussian_noise, 0, 255)

    def store_history(self):
        if len(self.history) >= 5:  # Keep only the last 5 states
            self.history.pop(0)
        self.history.append((np.copy(self.current_image), np.copy(self.current_target_image)))

    def undo_last_change(self, event):
        if self.history:
            self.current_image, self.current_target_image = self.history.pop()
            self.update_image()

    def update_image(self):
        self.ax.imshow(self.current_image, cmap='gray')
        self.fig.canvas.draw()

    def save_image(self, event):
        np.save(self.output_file_path, self.current_image.astype(self.original_image.dtype))
        np.save(self.output_target_file_path, self.current_target_image.astype(self.original_target_image.dtype))
        print(f"Images saved to {self.output_file_path} and {self.output_target_file_path}")
        plt.close(self.fig)

# Example usage
"""input_file_path = 'npy_img/L506_0_input.npy'
target_file_path = 'npy_img/L506_0_target.npy'

result_input_file_path = 'npy_img_sim/L506_0_input.npy'
result_target_file_path = 'npy_img_sim/L506_0_target.npy'

percentage = 0.3  # Gaussian amplitude as 50% of the brightest spot
sigma = 5  # Gaussian sigma

editor = ImageEditor(input_file_path, target_file_path, result_input_file_path, result_target_file_path, sigma = sigma, percentage=percentage)
plt.show()"""


# Parameters
source_dir = 'lowdose_simulation/npy_img_physics_simulation/'  # Directory with original files
save_to_dir = 'lowdose_simulation/with_tumor'  # Directory to save modified files

sigma = 5
amplitude_percentage = 0.3

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
    editor = ImageEditor(input_file_path, target_file_path, result_input_path, result_target_path, sigma=sigma, percentage=amplitude_percentage)
    plt.show()
    print(f"Processed {base_name}")