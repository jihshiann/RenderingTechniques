import cv2
import numpy as np
import pandas as pd
import random
import pandas as pd
import os

# Function to extract constants from the first line of the PA table
def extract_constants(pa_table):
    # 從第一行（列名）中提取 N, M, Z
    header_row = pa_table.columns
    n = int(header_row[1])  # N
    M = int(header_row[2])  # M
    Z = int(header_row[-1])  # Z

    # 從第二行中提取 weights
    weights_row = pa_table.iloc[0]  # 第二行的數據
    weights = np.array(weights_row[-4:-1]).astype(int)  # 取最後三個值作為 weights

    return n, M, weights, Z

 # Generates a secret message as a list of numbers in the given base
def generate_secretMsg(base, length):

    random.seed(100)
    return [random.randint(0, base - 1) for _ in range(length)]

def find_adjustment_values(pa_table, difference):

    adjustment_values = pa_table.loc[difference+1, ['w1', 'w2', 'w3']].values.astype(int)
    return adjustment_values

def correct_pixel_overflow(pixels, max_variation, weights, modulo, secret_value, pa_table):
    overflow_indices = np.where(pixels > 255)[0]
    for index in overflow_indices:
        pixels[index] -= max_variation
        pixels = update_pixel_values(pixels, weights, modulo, secret_value, pa_table)
    return pixels

def correct_pixel_underflow(pixels, max_variation, weights, modulo, secret_value, pa_table):
    underflow_indices = np.where(pixels < 0)[0]
    for index in underflow_indices:
        pixels[index] += max_variation
        pixels = update_pixel_values(pixels, weights, modulo, secret_value, pa_table)
    return pixels

def update_pixel_values(pixels, weights, modulo, secret_value, pa_table):
    remainder = np.dot(pixels, weights.transpose()) % modulo
    difference = (secret_value - remainder) % modulo
    adjustment = find_adjustment_values(pa_table, difference)
    return pixels + adjustment

def embed_secret_message(pixel, weights, modulo, pa_table, max_variation, secret_value):
    remainder = np.dot(pixel, weights.transpose()) % modulo
    difference = (secret_value - remainder) % modulo
    adjustment = find_adjustment_values(pa_table, difference)
    updated_pixel = pixel + adjustment

    updated_pixel = correct_pixel_overflow(updated_pixel, max_variation, weights, modulo, secret_value, pa_table)
    updated_pixel = correct_pixel_underflow(updated_pixel, max_variation, weights, modulo, secret_value, pa_table)
    
    return updated_pixel




# main
# Path to your PA Table file
current_directory = os.getcwd()
patab_folder = os.path.join(current_directory, 'patab')
patab_files = os.listdir(patab_folder)
patab_files = [
    'PA_3_6_(1_2_3)_1.csv',
    'PA_3_35_(1_11_16)_2.csv',
    'PA_3_87_(1_5_25)_2.csv'
]

# 讀取影像檔案
cover_folder = os.path.join(current_directory, 'cover')

# 逐圖執行
for index, filename in enumerate(os.listdir(cover_folder)):
   if filename.endswith(('.png')):
    # read img
    image_path = os.path.join(cover_folder, filename)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Extract constants from the PA table
    pa_cat = index //4
    patab_filename = patab_files[pa_cat]
    pa_table_path = os.path.join(patab_folder, patab_filename)
    pa_table = pd.read_csv(pa_table_path)
    num, M_ary, weights, max_variation = extract_constants(pa_table)
    
    # GWM
    unembedded_img = image.copy();
    embedded_img = image.copy();
    # 灰階
    if len(unembedded_img.shape) == 2:
        unembedded_img = unembedded_img.reshape(1, -1)
        embedded_img = embedded_img.reshape(1, -1)
        secretMsgs = generate_secretMsg(M_ary, int(unembedded_img.shape[1] / num))
        for i in range(0, unembedded_img.shape[1], num):
            cover_pixelGroup = np.array(unembedded_img[0,i:i+num])
            if cover_pixelGroup.shape[0] == 3:
                msg = secretMsgs[int(i / num)]
                updated_pixel = embed_secret_message(cover_pixelGroup, weights, M_ary, pa_table, max_variation, msg)
                embedded_img[0,i:i+num] = updated_pixel
    #彩色
    else:
        secretMsgs = generate_secretMsg(M_ary, unembedded_img.shape[0] * unembedded_img.shape[1])
        for i in range(unembedded_img.shape[0]):
            for j in range(unembedded_img.shape[1]):
                msg = secretMsgs[(i * unembedded_img.shape[1]) + j]
                updated_pixel = embed_secret_message(unembedded_img[i][j], weights, M_ary, pa_table, max_variation, msg)
                embedded_img[i, j] = updated_pixel
    
    #檔名
    base_name, extension = os.path.splitext(filename)
    save_name = (base_name + 
                 '_stego_N' + str(num) + '_M' + str(M_ary) + '_' + str(weights[0]) + '_' + str(weights[1]) + '_' + str(weights[2]) + '_Z' + str(max_variation) 
                 + extension)
    # 存檔
    save_path = os.path.join(current_directory, 'stego', save_name)
    cv2.imwrite(save_path, image)
  
        


