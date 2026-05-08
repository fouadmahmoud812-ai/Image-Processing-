import numpy as np
import cv2

def data_generator(df, batch_size):
    rgb_paths = df['rgb_file'].values
    depth_paths = df['depth_file'].values
    calories = df['calories'].values
    num_samples = len(df)

    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        batch_rgb, batch_depth, batch_y = [], [], []

        for idx in indices:
            try:
                img = cv2.imread(rgb_paths[idx])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img = img.astype(np.float32) / 255.0

                depth = cv2.imread(depth_paths[idx], cv2.IMREAD_GRAYSCALE)
                depth = cv2.resize(depth, (224, 224))


                depth = cv2.medianBlur(depth, 5)

                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                depth = cv2.filter2D(depth, -1, kernel)

                sobelx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
                depth_edges = cv2.magnitude(sobelx, sobely)

                depth = cv2.addWeighted(depth.astype(np.float32), 0.7, depth_edges.astype(np.float32), 0.3, 0)

                depth = depth / (np.max(depth) + 1e-6)
                depth = np.expand_dims(depth, axis=-1)


                if np.random.rand() > 0.5:
                    img, depth = np.fliplr(img), np.fliplr(depth)
                if np.random.rand() > 0.5:
                    img, depth = np.flipud(img), np.flipud(depth)

                batch_rgb.append(img)
                batch_depth.append(depth)
                batch_y.append(calories[idx])

                if len(batch_rgb) == batch_size:
                    yield ((np.array(batch_rgb), np.array(batch_depth)), np.array(batch_y))
                    batch_rgb, batch_depth, batch_y = [], [], []

            except Exception as e:
                continue
