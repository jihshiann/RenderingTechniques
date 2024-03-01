import pandas as pd
import numpy as np
import random
import copy
import cv2
import os
import re


#矩形轉換
def rectangular_transform(image, a, b, c, d, M, N):
    image_change = copy.deepcopy(image)
    A = np.array([[a, b], [c, d]])
    for i in range(N):
        for j in range(M):
            pixel_change = np.mod(np.dot(A, np.array([[j], [i]])), np.array([[M], [N]]))
            image_change[pixel_change[1][0], pixel_change[0][0]] = image[i, j]

    return image_change

#從文件中讀取RT秘鑰
def read_RT_secret_key(file_path):
    with open(file_path, 'r') as f:
        line = f.readline().split(' ')
        a, b, c, d, M, N = [int(x) for x in line[:6]]
        G = int(f.readline())

    return a, b, c, d, M, N, G

#圖像頻道排列
def channel_permutation(image, PK):
    image_change = copy.deepcopy(image)
    if PK == 1:
        image_change[:, :, [1, 2]] = image[:, :, [2, 1]]
    elif PK == 2:
        image_change[:, :, [0, 1]] = image[:, :, [2, 0]]
    elif PK == 3:
        image_change[:, :, [0, 2]] = image[:, :, [1, 2]]
    elif PK == 4:
        image_change[:, :, [0, 2]] = image[:, :, [1, 0]]
    elif PK == 5:
        image_change[:, :, [0, 1]] = image[:, :, [2, 1]]

    return image_change

#將三張灰階影像合成一張彩色影像
def channel_composition(image1, image2, image3):
    return np.stack([image3.flatten(), image2.flatten(), image1.flatten()], axis=1).reshape((image1.shape[0], image1.shape[1], 3))

#處理像素點
def process_pixel(pixel, Z, n, W, M):
    if pixel >= 0 and pixel < Z:
        P = np.array([Z] * n)
    elif pixel > (255 - Z) and pixel <= 255:
        P = np.array([255 - Z] * n)
    else:
        P = np.array([pixel] * n)

    r = np.dot(P, W.T) % M
    return P, r

def find_Ad(RPA_table, d):
    A_d = RPA_table.loc[RPA_table['d'] == d].to_numpy()
    return A_d[0, 1:]

def save_secret_messages_to_file(messages, seed):
    with open(f"11-mesmea/mes_mea_{int(seed/100)}.txt", "w") as mesmea_file:
        for message in messages:
            mesmea_file.write(f"{message} ")

#產生密訊
def generate_secret_messages(M, size, seed):
    random.seed(seed)
    secret_messages = [random.randint(0, 100) % M for _ in range(size)]

    save_secret_messages_to_file(secret_messages, seed)
    return secret_messages

def GMWRDH(image, RPA_table, n, M, W, Z, seed):
    image_change1, image_change2, image_change3 = copy.deepcopy(image), copy.deepcopy(image), copy.deepcopy(image)
    secret_messages = generate_secret_messages(M, image.shape[0] * image.shape[1], seed)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            P, r = process_pixel(image[i, j], Z, n, W, M)
            S = secret_messages[(i * image.shape[1]) + j]
            d = (S - r) % M
            A_d = find_Ad(RPA_table, d)
            P_prime = P + A_d

            image_change1[i][j], image_change2[i][j], image_change3[i][j] = P_prime[0], P_prime[1], P_prime[2]
        
    return image_change1, image_change2, image_change3

def extract_numbers_from_filename(filename):
    return [int(number) for number in re.findall(r'\d+', filename)]

# RPA表取參數
def read_RPA_table(RPA_table_file_name):
    RPA_table = pd.read_csv(f'10-rpatab/{RPA_table_file_name}')
    RPA_table = RPA_table.drop(columns=[RPA_table.columns[0], RPA_table.columns[2], RPA_table.columns[6]], axis=1)
    RPA_table = RPA_table.iloc[:-3, :]
    columns_name = RPA_table.iloc[0].tolist()
    RPA_table.columns = columns_name
    RPA_table = RPA_table.drop(RPA_table.index[0])
    RPA_table = RPA_table.reset_index(drop=True).astype(int)

    numbers = extract_numbers_from_filename(RPA_table_file_name)
    n, M, W, Z = numbers[0], numbers[1], np.array(numbers[2:-1]).astype(int), numbers[-1]

    return RPA_table, n, M, W, Z

#處理資料夾所有圖像
def process_images(folder_path):
    image_files = os.listdir(folder_path)
    RPA_table_files = os.listdir("10-rpatab")
    permutation_keys = np.random.randint(0, 6, size=len(image_files))

    for index, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        RPA_table, n, M, W, Z = read_RPA_table(RPA_table_files[index])

        image_change1, image_change2, image_change3 = GMWRDH(image, RPA_table, n, M, W, Z, (index + 1) * 100)

        # 生成轉換後的圖片檔名
        image_new_name1 = f'{image_file[:-4]}_mark_N{n}_M{M}_{W[0]}_{W[1]}_{W[2]}_Z{Z}_I1{image_file[-4:]}'
        image_new_name2 = f'{image_file[:-4]}_mark_N{n}_M{M}_{W[0]}_{W[1]}_{W[2]}_Z{Z}_I2{image_file[-4:]}'
        image_new_name3 = f'{image_file[:-4]}_mark_N{n}_M{M}_{W[0]}_{W[1]}_{W[2]}_Z{Z}_I3{image_file[-4:]}'

        # 保存處理後的圖片
        cv2.imwrite(f'2-marked/{image_new_name1}', image_change1)
        cv2.imwrite(f'2-marked/{image_new_name2}', image_change2)
        cv2.imwrite(f'2-marked/{image_new_name3}', image_change3)

        # 將三張灰階影像合成彩色影像
        gray_to_color_image = channel_composition(image_change1, image_change2, image_change3)

        # 生成彩色影像的檔名
        image_new_name4 = f'{image_file[:-4]}_channe_N{n}_M{M}_{W[0]}_{W[1]}_{W[2]}_Z{Z}{image_file[-4:]}'

        # 保存彩色影像
        cv2.imwrite(f'3-channe/{image_new_name4}', gray_to_color_image)

        # 進行通道排列
        channel_permutation_image = channel_permutation(gray_to_color_image, permutation_keys[index])

        # 生成排列後的圖片檔名
        image_new_name5 = f'{image_file[:-4]}_permut_N{n}_M{M}_{W[0]}_{W[1]}_{W[2]}_Z{Z}{image_file[-4:]}'

        # 保存排列後的圖片
        cv2.imwrite(f'4-permut/{image_new_name5}', channel_permutation_image)

        # 讀取RT秘密鑰匙
        a, b, c, d, M, N, G = read_RT_secret_key(f'12-encpar/{image_file[:-4]}-Secret-Key.txt')

        # 進行RT轉換
        for _ in range(G):
            channel_permutation_image = rectangular_transform(channel_permutation_image, a, b, c, d, M, N)

        # 生成加密後的圖片檔名
        encrypted_image_name = f'{image_file[:-4]}_enc{image_file[-4:]}'

        # 保存加密後的圖片
        cv2.imwrite(f'5-encry/{encrypted_image_name}', channel_permutation_image)


#MAIN
np.random.seed(24)
folder_path = "1-origin"
process_images(folder_path)

