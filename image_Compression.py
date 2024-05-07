from time import sleep
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
import threading

def load_image_thread():
    """启动后台线程来加载图像，避免UI冻结，并更新状态信息。"""
    threading.Thread(target=load_image).start()

def load_image():
    """通过文件对话框加载图像并显示在GUI中，同时保持纵横比，以避免UI冻结。"""
    global original_image, display_image
    file_path = filedialog.askopenfilename()  # 打开文件对话框，选择图像文件
    if not file_path:
        return
    
    # 清空旧的PCA和SVD信息
    pca_info.set("")
    svd_info.set("")
    status_label.config(text="正在加载图像...")
    
    img = Image.open(file_path)  # 打开图像文件
    original_image = np.array(img)  # 将图像转换为NumPy数组，保存原始全尺寸图像用于压缩
    img.thumbnail((400, 400), Image.Resampling.LANCZOS)  # 缩小版本用于显示
    display_image = ImageTk.PhotoImage(img)  # 创建用于显示的图像对象
    original_image_label.config(image=display_image)  # 在GUI中显示原始图像
    original_image_label.image = display_image
    calculate_info(original_image)  # 计算并显示每个颜色通道的PCA和SVD奇异值总数
    status_label.config(text="加载完成。")

def pca_compress(image, k):
    """使用PCA对彩色图像的每个通道进行压缩和重建."""
    channels = [image[:, :, i] for i in range(3)]  # 将彩色图像的每个通道分离出来
    compressed_channels = []
    for channel in channels:
        mean_centered = channel - np.mean(channel, axis=0)  # 对每个通道进行均值中心化
        covariance_matrix = np.cov(mean_centered, rowvar=False)  # 计算协方差矩阵
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)  # 计算特征值和特征向量
        sorted_index = np.argsort(eigenvalues)[::-1]  # 对特征值进行降序排序
        eigenvectors = eigenvectors[:, sorted_index]  # 对特征向量进行排序
        selected_eigenvectors = eigenvectors[:, :k]  # 选择前k个特征向量
        transformed = np.dot(mean_centered, selected_eigenvectors)  # 对均值中心化后的数据进行变换
        reconstructed = np.dot(transformed, selected_eigenvectors.T) + np.mean(channel, axis=0)  # 重建图像
        compressed_channels.append(reconstructed)
    compressed_image = np.stack(compressed_channels, axis=2)  # 将压缩后的通道合并为彩色图像
    return compressed_image.astype(np.uint8)

def svd_compress(image, k):
    """使用SVD对彩色图像的每个通道进行压缩和重建."""
    channels = [image[:, :, i] for i in range(3)]  # 将彩色图像的每个通道分离出来
    compressed_channels = []
    for channel in channels:
        U, S, VT = np.linalg.svd(channel, full_matrices=False)  # 进行奇异值分解
        S = np.diag(S[:k])  # 选择前k个奇异值
        U = U[:, :k]  # 选择前k个左奇异向量
        VT = VT[:k, :]  # 选择前k个右奇异向量
        compressed_channel = np.dot(U, np.dot(S, VT))  # 重建通道
        compressed_channels.append(compressed_channel)
    compressed_image = np.stack(compressed_channels, axis=2)  # 将压缩后的通道合并为彩色图像
    return compressed_image.astype(np.uint8)

def sequential_compress(image, k_pca, k_svd):
    """先用PCA压缩，然后在结果上应用SVD压缩."""
    pca_compressed = pca_compress(image, k_pca)  # 使用PCA压缩图像
    svd_compressed = svd_compress(pca_compressed, k_svd)  # 在PCA压缩结果上应用SVD压缩
    return svd_compressed    

def calculate_info(image):
    """计算并显示每个颜色通道的PCA和SVD奇异值总数，并更新状态。"""
    status_label.config(text="正在计算奇异值...")
    channels = [image[:, :, i] for i in range(3)]  # 将彩色图像的每个通道分离出来
    pca_counts, svd_counts = [], []
    for idx, channel in enumerate(channels):
        mean_centered = channel - np.mean(channel, axis=0)  # 对每个通道进行均值中心化
        covariance_matrix = np.cov(mean_centered, rowvar=False)  # 计算协方差矩阵
        eigenvalues, _ = np.linalg.eigh(covariance_matrix)  # 计算特征值
        non_zero_eigenvalues = np.sum(eigenvalues > np.finfo(float).eps)  # 统计非零特征值的数量
        pca_counts.append(non_zero_eigenvalues)

        _, S, _ = np.linalg.svd(channel, full_matrices=False)  # 进行奇异值分解
        svd_counts.append(len(S))
    
    pca_info.set(f"PCA components (R,G,B): {pca_counts}")  # 显示每个颜色通道的PCA奇异值总数
    svd_info.set(f"SVD singular values (R,G,B): {svd_counts}")  # 显示每个颜色通道的SVD奇异值总数
    status_label.config(text="计算完成。")

def compress_and_display_thread():
    """启动后台线程来压缩图像，避免UI冻结，并更新状态信息。"""
    threading.Thread(target=compress_and_display).start()

def compress_and_display():
    global compressed_img  # 将压缩后的图像设置为全局变量
    if original_image is None:
        messagebox.showerror("Error", "No image loaded.")
        return
    method = method_var.get().lower()  # 获取选择的压缩方法
    k = simpledialog.askinteger("Input", f"Enter the number of components to retain for {method.upper()}:", minvalue=1, maxvalue=original_image.shape[1])  # 获取保留的特征数量
    if k is None:
        return   
    status_label.config(text="正在压缩...")
    root.update_idletasks()  # 强制更新界面
    if method == 'pca':
        compressed_img = pca_compress(original_image, k)  # 使用PCA压缩图像
    elif method == 'svd':
        compressed_img = svd_compress(original_image, k)  # 使用SVD压缩图像
    elif method == 'sequential':
        k_svd = simpledialog.askinteger("Input", "Enter the number of SVD singular values to retain for the second step:", minvalue=1, maxvalue=original_image.shape[1])  # 获取保留的奇异值数量
        if k_svd is None:
            return
        compressed_img = sequential_compress(original_image, k, k_svd)  # 先用PCA压缩，然后在结果上应用SVD压缩
    else:
        messagebox.showerror("Error", "Invalid compression method.")
        status_label.config(text="压缩失败：无效的压缩方法")
        return   
    img = Image.fromarray(compressed_img)  # 将NumPy数组转换为图像对象
    img.thumbnail((400, 400), Image.Resampling.LANCZOS)  # 缩小版本用于显示
    photo = ImageTk.PhotoImage(img)  # 创建用于显示的图像对象
    compressed_image_label.config(image=photo)  # 在GUI中显示压缩后的图像
    compressed_image_label.image = photo
    status_label.config(text="压缩完成。")

def clear_compressed_image():
    """清空压缩图片显示。"""
    compressed_image_label.config(image='')
    compressed_image_label.image = None

def save_compressed_image():
    """保存压缩后的图像。"""
    global compressed_img
    if compressed_img is None:
        messagebox.showerror("Error", "No compressed image to save.")
        return
    img = Image.fromarray(compressed_img)  # 将NumPy数组转换为图像对象
    save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"),("All files", "*.*")])  # 打开文件对话框，选择保存路径
    if not save_path:
        return
    img.save(save_path)  # 保存图像到文件
    messagebox.showinfo("Success", "Compressed image saved successfully.")

root = tk.Tk()
root.title("Image Compression Tool")
root.geometry("1024x600")

original_image = None
compressed_img = None
pca_info = tk.StringVar(root)
svd_info = tk.StringVar(root)

frame = tk.Frame(root)
frame.pack(pady=20)

load_button = tk.Button(frame, text="Load Image", command=load_image_thread)
load_button.pack(side=tk.LEFT, padx=10)

method_var = tk.StringVar(root)
method_var.set("PCA")
method_option = tk.OptionMenu(frame, method_var, "PCA", "SVD", "Sequential")
method_option.pack(side=tk.LEFT, padx=10)

compress_button = tk.Button(frame, text="Compress", command=compress_and_display)
compress_button.pack(side=tk.LEFT, padx=10)

clear_button = tk.Button(frame, text="Clear Compressed Image", command=clear_compressed_image)
clear_button.pack(side=tk.LEFT, padx=10)

save_button = tk.Button(frame, text="Save Compressed Image", command=save_compressed_image)
save_button.pack(side=tk.LEFT, padx=10)

info_label_pca = tk.Label(root, textvariable=pca_info)
info_label_pca.pack()

info_label_svd = tk.Label(root, textvariable=svd_info)
info_label_svd.pack()

status_label = tk.Label(root, text="就绪")
status_label.pack(side="bottom", fill="x")

original_frame = tk.Frame(root)
original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

original_label = tk.Label(original_frame, text="Original Image")
original_label.pack()

original_image_label = tk.Label(original_frame)
original_image_label.pack()

compressed_frame = tk.Frame(root)
compressed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

compressed_label = tk.Label(compressed_frame, text="Compressed Image")
compressed_label.pack()

compressed_image_label = tk.Label(compressed_frame)
compressed_image_label.pack()

root.mainloop()