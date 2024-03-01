import csv
import itertools
import math
import os
from pickletools import uint8
import numpy as np
import cv2
import time
from PIL import Image



## 讀取影像檔案
current_directory = os.getcwd()
source_folder = os.path.join(current_directory, 'source')
decryp_folder = os.path.join(current_directory, 'decryp')
encryp_folder = os.path.join(current_directory, 'encryp')
### 檔名處理
for filename in os.listdir(source_folder):
   if filename.endswith(('.png')):
        image_path = os.path.join(source_folder, filename)
        file_name, file_extension = os.path.splitext(filename)
        enc_file_name = file_name + '_enc' + file_extension
        enc_path = os.path.join(encryp_folder, enc_file_name)
        dec_file_name = file_name + '_dec' + file_extension
        dec_path = os.path.join(decryp_folder, dec_file_name)
        key_path = os.path.join(encryp_folder, file_name + '-Secret-Key.txt')
        decTime_path = os.path.join(decryp_folder, file_name + '_dec_time.txt')
        transformed_image = cv2.imread(enc_path, cv2.IMREAD_UNCHANGED)

        with open(key_path, 'r') as file:
            lines = file.readlines()
        
        a = int(lines[0])
        b = int(lines[1])
        c = int(lines[2])
        d = int(lines[3])
        m = int(lines[4])
        n = int(lines[5])
        g = math.floor(int(lines[6])/2)
        matrix = [[a, b], [c, d]]
     
        
        start = time.time()
        
        # 初始化還原後的圖像
        restored_image = np.array(transformed_image)

        for i in range(g):
            # 創建一個臨時圖像用於這次迭代的轉換
            temp_image = np.zeros_like(restored_image)
            for y in range(n):
                for x in range(m):
                    original_pos = np.array([x, y]).T
                    new_pos = np.dot(original_pos, matrix) % np.array([m, n]).T
                    x_prime, y_prime = int(new_pos[0]), int(new_pos[1])
                    temp_image[x_prime, y_prime] = restored_image[x, y]

            # 更新轉換後的圖像為這次迭代的結果
            restored_image = temp_image
            print(i)

        # 檢查是否可逆
        # det = np.linalg.det(matrix)
        # if det == 0:
        #     raise ValueError("轉換矩陣不可逆")

        # inverse_matrix = np.linalg.inv(matrix)

        # # 逆轉換
        # # 初始化恢復後的圖像
        # restored_image = np.array(transformed_image)
        # for i in range(g):
        #     # 創建一個臨時圖像用於這次迭代的逆轉換
        #     temp_image = np.zeros_like(restored_image)
        #     for y in range(n):
        #         for x in range(m):
        #             new_pos = np.dot(inverse_matrix, np.array([x, y]).T) % np.array([m, n]).T
        #             x_ori, y_ori = int(new_pos[0]), int(new_pos[1])
        #             #temp_image[x_ori, y_ori] = restored_image[x, y]
        #             restored_image[x_ori, y_ori] = restored_image[x, y]

        #     # 更新恢復後的圖像為這次迭代的結果
        #     #restored_image = temp_image


        end = time.time()
        # 計算經過的時間並格式化到小數點後兩位
        elapsed_time = "{:.2f}".format(end - start)

        # 準備要寫入檔案的文字
        text = "G: {}\ntime: {} 秒".format(g, elapsed_time)

        # 寫入檔案
        with open(decTime_path, 'w', encoding='utf-8') as file:
            file.write(text)
        # print("執行時間：%.2f 秒" % (elapsed_time))
        
        cv2.imwrite(dec_path, restored_image)


