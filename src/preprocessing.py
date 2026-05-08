import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocessing(rgb_path, depth_path):

    img_bgr = cv2.imread(rgb_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    img_res = cv2.resize(img_rgb, (224, 224))
    depth_res = cv2.resize(depth_raw, (224, 224))

    depth_filtered = cv2.medianBlur(depth_res, 5)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    depth_sharpened = cv2.filter2D(depth_filtered, -1, kernel)


    edges_canny = cv2.Canny(depth_filtered, 50, 150)

    sobelx = cv2.Sobel(depth_filtered, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(depth_filtered, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)


    _, mask = cv2.threshold(depth_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    img_norm = img_res / 255.0
    img_flipped = np.fliplr(img_norm)


    titles = [
        '1. Original RGB', '2. Raw Depth', '3. Depth Filtering',
        '4. Sharpened Depth', '5. Canny Edges', '6. Sobel (Gradient)',
        '7. Segmentation Mask', '8. Normalized RGB', '9. Augmented (Flip)'
    ]

    images = [
        img_rgb, depth_raw, depth_filtered,
        depth_sharpened, edges_canny, sobel_combined,
        mask, img_norm, img_flipped
    ]

    plt.figure(figsize=(18, 15))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        is_gray = i in [1, 2, 3, 4, 5, 6]
        plt.imshow(images[i], cmap='gray' if is_gray else None)
        plt.title(titles[i], fontsize=14, fontweight='bold')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
