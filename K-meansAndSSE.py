import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from PIL import Image

# Thay tên file ảnh vào biến p1 (chỉ tên file, không có đuôi .jpg)
p1 = "champon-g31fa88e14_640"

png_image_path = p1 + ".jpg"
png_image = Image.open(png_image_path)

jpg_image_path = p1 + ".png"
png_image.convert("RGB").save(jpg_image_path, "PNG")
img = plt.imread(jpg_image_path)

sse = []
width = img.shape[0]
height = img.shape[1]
img = img.reshape(width * height, 3)

# Áp dụng PCA để giảm chiều dữ liệu
n_components_range = range(1, min(width, height, 3) + 1)  # Lựa chọn một dải giá trị hợp lý
for n_components in n_components_range:
    pca = PCA(n_components=n_components)
    img_pca = pca.fit_transform(img)

    k = 0
    while True:
        k += 1
        kmeans = KMeans(n_clusters=k).fit(img_pca)
        sse.append(kmeans.inertia_)
        print(k)
        if sse[k - 1] <= 2000:
            break

    # Hiển thị kết quả của PCA và Elbow Method
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, k + 1), sse, marker='o')
    plt.title(f'Elbow method with PCA (n_components={n_components})')
    plt.xlabel('Number of clusters')
    plt.ylabel('Errors (Total distance)')
    plt.show()

    # Tìm giá trị n_components tốt nhất (bạn có thể chọn một cách thủ công)
    if n_components == 2:  # Đây chỉ là một ví dụ, bạn có thể chọn giá trị phù hợp hơn
        break
