import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

dataset_path = 'data/APPLE_DISEASE_DATASET'
classes = os.listdir(dataset_path)

# Display 3 random images from each class
for cls in classes:
    class_path = os.path.join(dataset_path, cls)
    images = os.listdir(class_path)
    sample_images = random.sample(images, 3)

    for img_name in sample_images:
        img_path = os.path.join(class_path, img_name)
        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.title(cls)
        plt.axis('off')
        plt.show()
