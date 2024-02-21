import os
import re
import cv2
import copy
import random
import numpy as np
import pandas as pd



def find_Ad(RPA_table, d):
    # 在 RPA_table 中尋找符合條件的行：列 'd' 的值等於參數 d
    filtered_row = RPA_table[RPA_table['d'] == d]

    # 將找到的行轉換為 NumPy 數組格式
    Ad_array = filtered_row.to_numpy()

    # 獲取數組的第一行，但不包括第一列（即 'd' 列）
    Ad = Ad_array[0, 1:]

    return Ad


def generate_secretMsg(max_value, message_count, seed_value):
    # 初始化
    random.seed(seed_value)

    # 生成隨機數當秘文
    secret_messages = []
    for _ in range(message_count):
        message = random.randint(0, 100) % max_value
        secret_messages.append(message)

    # 存密文
    filename = f"mesmea/mes_mea_{int(seed_value / 100)}.txt"
    with open(filename, "w") as file:
        for message in secret_messages:
            file.write(f"{message} ")

    return secret_messages


def exec_GMWRDH(original_image, RPA_table, n, M, W, Z, seed):
    image_change1 = copy.deepcopy(original_image)
    image_change2 = copy.deepcopy(original_image)
    image_change3 = copy.deepcopy(original_image)

    # 生成祕文
    secret_messages = generate_secretMsg(M, original_image.shape[0] * original_image.shape[1], seed) 

    imgs_changed = []
    # 遍歷每個像素
    for i in range(original_image.shape[0]):
        for j in range(original_image.shape[1]):
            # 根據像素值選擇參數 P
            pixel_value = original_image[i][j]
            if pixel_value >= 0 and pixel_value < Z:
                P = np.array([Z] * n)
            elif pixel_value > (255 - Z) and pixel_value <= 255:
                P = np.array([255 - Z] * n)
            else:
                P = np.array([pixel_value] * n)

            r = np.dot(P, W.T) % M
            S = secret_messages[(i * original_image.shape[1]) + j]
            d = (S - r) % M
            Ad = find_Ad(RPA_table, d)
            P_prime = P + Ad

            # 更新三個更改後的圖像
            image_change1[i][j] = P_prime[0]
            image_change2[i][j] = P_prime[1]
            image_change3[i][j] = P_prime[2]

    imgs_changed.extend([image_change1, image_change2, image_change3])

    return imgs_changed


def extract_constants(rpaTable_filename):
    rpaTable_path = f'rpatab/{rpaTable_filename}'
    RPA_table = pd.read_csv(rpaTable_path)

    # 移除列
    columns_to_drop = [RPA_table.columns[0], RPA_table.columns[2], RPA_table.columns[6]]
    RPA_table = RPA_table.drop(columns=columns_to_drop, axis=1)
    # 移除末尾的幾行
    RPA_table = RPA_table.iloc[:-3, :]
    # 列名
    columns_name = RPA_table.iloc[0].tolist()
    RPA_table.columns = columns_name
    # 移除包含列名的行並重置索引
    RPA_table = RPA_table.drop(RPA_table.index[0])
    RPA_table = RPA_table.reset_index(drop=True).astype(int)

    # 從檔名取常數 n, M, W, Z
    constants_in_filename = re.findall(r'\d+', rpaTable_filename)
    n = int(constants_in_filename[0])
    M = int(constants_in_filename[1])
    W = np.array(constants_in_filename[2:-1]).astype(int)
    Z = int(constants_in_filename[-1])

    return RPA_table, n, M, W, Z



# main
current_directory = os.getcwd()
rpatab_folder = os.path.join(current_directory, 'rpatab')
rpatab_files = os.listdir(rpatab_folder)
origin_folder = os.path.join(current_directory, 'origin')
origin_files = os.listdir(origin_folder)


# 逐圖執行
for index, filename in enumerate(origin_files):
    # read img
    image_path = os.path.join(origin_folder, filename)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Extract constants from the RPA table
    rpaTable, N, M, W, Z = extract_constants(rpatab_files[origin_files.index(filename)])

    # GMWRDH ALGO
    imgs_changed = exec_GMWRDH(image, rpaTable, N, M, W, Z, (origin_files.index(filename) + 1) * 100)
    
    # SAVE
    for no, img_changed in enumerate(imgs_changed):
        #檔名
        base_name, extension = os.path.splitext(filename)
        save_name = (base_name + 
                     '_mark_N' + str(N) + '_M' + str(M) + '_' + str(W[0]) + '_' + str(W[1]) + '_' + str(W[2]) + '_Z' + str(Z) + '_I' + str(no+1)
                     + extension)
    
        # 存檔
        save_path = os.path.join(current_directory, 'marked', save_name)
        cv2.imwrite(save_path, img_changed)
        
print('END')