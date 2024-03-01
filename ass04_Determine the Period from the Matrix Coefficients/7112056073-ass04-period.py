import csv
import itertools
import math
import os
import numpy as np


def is_valid_ratio(M, N):
    # 檢查 M/N 是否為整數，或 N/M 是否為奇數
    return (M % N == 0) or (N % M == 1)

def generate_transformations(M, N, a, b, c, d):
    print(f"'M', 'N', 'a', 'b', 'c', 'd', 'period'")
    transformations = []
    
    # 計算 p, l1, l2
    p = math.gcd(M, N)
    l1 = M // p
    l2 = N // p
    num = 0


    # 檢查組合是否符合給定的條件
    if ((b % l1 == 0) or (c % l2 == 0)) and \
        (math.gcd(a * d - b * c, p) == 1) and \
        (math.gcd(a, l1) == 1) and (math.gcd(d, l2) == 1):
            
        matrix = [[a, b], [c, d]]
        period = 0
        x = np.arange(0, M)
        y = np.arange(0, N)
        pixels = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        new_pixels = np.dot(pixels, matrix) 
        new_pixels[:, 0] %= M 
        new_pixels[:, 1] %= N
        period += 1
        while not np.all(new_pixels == pixels):
            new_pixels = np.dot(new_pixels, matrix) 
            new_pixels[:, 0] %= M
            new_pixels[:, 1] %= N 
            period += 1
            # try到M*N/2放棄，但3*3不符合
            if period > M*N/2:
                period = 0
                break
                
        num += 1
        transformations.append([M, N, a, b, c, d, period])
        print(f"{M}, {N}, {a}, {b}, {c}, {d}, {period}")

    return transformations


def save_to_csv(M, N, a, b, c, d, data):
    # 將數據保存到CSV文件中
    filename = "ass04-result.csv"

    if not os.path.exists(filename):
        mode = 'w'  # 如果文件不存在，則創建並寫入
        last_no = 0  # 從0開始
    else:
        mode = 'a'  # 如果文件已存在，則追加
        # 需要讀取最後一行來獲得最後的 "No."
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            last_row = None
            for last_row in reader: pass  # 遍歷到最後一行
            last_no = int(last_row[0]) if last_row else 0  # 如果文件不是空的，獲得最後的 "No." 並轉為整數

    with open(filename, mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        if mode == 'w':
            # 寫入欄位名稱
            writer.writerow(['No', 'M', 'N', 'a', 'b', 'c', 'd', 'period'])
    
        # 假設 data 是一個已經準備好的列表的列表，每個子列表代表一行
        # 更新每行的 "No." 值，然後寫入文件
        for row in data:
            last_no += 1  # 增加 "No." 值
            writer.writerow([last_no] + row)

    print(f"Data has been written to {filename}")

def main():
    try:
        # 讀取影像解析度和轉換矩陣範圍
        input_str = input("M, N: ")
        input1, input2 = input_str.split()
        M = int(input1)
        N = int(input2)

        input_str = input("a, b, c, d: ")
        input1, input2, input3, input4 = input_str.split()
        a = int(input1)
        b = int(input2)
        c = int(input3)
        d = int(input4)


        # 生成轉換矩陣
        transformations = generate_transformations(M, N, a, b, c, d)
        
        # 並保存到 CSV
        if transformations:
            save_to_csv(M, N, a, b, c, d, transformations)
        else:
            print("Invalid RT matrix!")

    except ValueError as e:
        print(f"ValueError {str(e)}")
    except Exception as e:
        print(f"Exception: {str(e)}")

main()

