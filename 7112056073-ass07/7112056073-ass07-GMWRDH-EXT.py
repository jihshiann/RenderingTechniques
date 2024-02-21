import re
import os
import cv2
import csv
import copy
import numpy as np
import pandas as pd


def calcul_ER(H, V, M):
    # 計算嵌入率
    return round(H * V * np.log2(M) / 3, 5)
def calcul_EC(H, V, M):
    # 計算嵌入容量
    return round(H * V * np.log2(M), 0)
def calcul_PSNR(MSE):
    # 計算峰值信噪比
    # 檢查 MSE 是否為零，以避免除以零的錯誤
    if MSE == 0:
        return float('inf')  # 如果 MSE 為零，則 PSNR 是無限大
    return round(10 * np.log10((255**2) / MSE), 2)
def calcul_MSE(image_new_name):
    # 計算均方誤差
    image_origin_name = '_'.join(image_new_name.split('_')[:1]) + '.png'
    image_origin = cv2.imread(f'origin/{image_origin_name}', cv2.IMREAD_UNCHANGED)
    image_new = cv2.imread(f'restor/{image_new_name}', cv2.IMREAD_UNCHANGED)

    # 檢查圖像是否成功加載
    if image_origin is None or image_new is None:
        print(f"無法加載圖像：{image_origin_name} 或 {image_new_name}")
        return -1  # 或者選擇引發一個異常

    return np.mean((image_origin - image_new) ** 2)


def decry_GMWRDH_img(image1, image2, image3, n):
    # 解密圖像
    image_origin = copy.deepcopy(image1)
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            P_prime = np.array([image1[i][j], image2[i][j], image3[i][j]])
            image_origin[i][j] = round(np.sum(P_prime) / n)
    return image_origin

def extract_GMWRDH_msg(image1, image2, image3, W, M, seed):
    # 取密文
    secret_messages = []
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            P_prime = np.array([image1[i][j], image2[i][j], image3[i][j]])
            S = np.dot(P_prime, W.T) % M
            secret_messages.append(S)
    
    # 寫密文
    with open(f"mesext/mes_ext_{int(seed/100)}.txt", "w") as mesext_file:
        mesext_file.write(' '.join(map(str, secret_messages)))
    
    return secret_messages

def read_rpaTable(RPA_table_file_name):
    RPA_table = pd.read_csv(f'rpatab/{RPA_table_file_name}')
    RPA_table = RPA_table.drop(columns=[RPA_table.columns[x] for x in [0, 2, 6]])
    RPA_table = RPA_table.iloc[:-3, :]
    RPA_table.columns = RPA_table.iloc[0].tolist()
    RPA_table = RPA_table.drop(RPA_table.index[0]).reset_index(drop=True).astype(int)

    # 從文件名提取常數
    RPA_table_name_numbers = re.findall(r'\d+', RPA_table_file_name)
    n, M, W, Z = int(RPA_table_name_numbers[0]), int(RPA_table_name_numbers[1]), np.array(RPA_table_name_numbers[2:-1]).astype(int), int(RPA_table_name_numbers[-1])
    return RPA_table, n, M, W, Z

def get_rpaTable_name(image_name):
    image_name_parts = image_name.split('_')
    image_name_numbers = re.findall(r'\d+', '_'.join(image_name_parts[2:]))
    RPA_rpaTable_name = 'RPA_' + image_name_numbers[0] + '_' + image_name_numbers[1] + '_(' + '_'.join(image_name_numbers[2:5]) + ')_' + image_name_numbers[5] + '.csv'
    return RPA_rpaTable_name



# MAIN
current_directory = os.getcwd()
marked_folder = os.path.join(current_directory, 'marked')
marked_files = np.reshape(os.listdir(marked_folder), (-1, 3))

for index, filenames in enumerate(marked_files):
    # 讀取圖像
    images = [cv2.imread(os.path.join(marked_folder, f), cv2.IMREAD_UNCHANGED) for f in filenames]
    
    # 讀取和處理 RPA 表格
    RPA_table, n, M, W, Z = read_rpaTable(get_rpaTable_name(filenames[0]))
    
    # 讀祕文並解密圖
    secret_messages = extract_GMWRDH_msg(*images, W, M, (index + 1) * 100)
    image_origin = decry_GMWRDH_img(*images, n)

    # 保存解密後的圖像
    save_name = '_'.join(filenames[0].split('_')[:2]) + '_rest.png'
    cv2.imwrite(os.path.join(current_directory, 'restor', save_name), image_origin)

    # 計算指標
    MSE = calcul_MSE(save_name)
    PSNR = calcul_PSNR(MSE)
    EC = calcul_EC(image_origin.shape[1], image_origin.shape[0], M)
    ER = calcul_ER(image_origin.shape[1], image_origin.shape[0], M)

    # 存檔
    csv_name = '_'.join(filenames[0].split('_')[:2]) + '_qualit.csv'
    with open(f'imgres/{csv_name}', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['RPA', str(n), str(M), 'w1', 'w2', 'w3', str(W[0]), str(W[1]), str(W[2])])
        writer.writerow(['MSE', str(MSE)])
        writer.writerow(['PSNR', str(PSNR)])
        writer.writerow(['EC', str(EC)])
        writer.writerow(['ER', str(ER)])

print('END')
