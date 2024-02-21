import csv
import itertools
import math
import numpy as np


def is_valid_ratio(M, N):
    # 檢查 M/N 是否為整數，或 N/M 是否為奇數
    return (M % N == 0) or (N % M == 1)

def generate_transformations(M, N, a1, a2, b1, b2, c1, c2, d1, d2):
    print(f"'No.', 'a', 'b', 'c', 'd', 'period'")
    transformations = []
    
    # 計算 p, l1, l2
    p = math.gcd(M, N)
    l1 = M // p
    l2 = N // p
    num = 0

    # 生成所有可能的 a, b, c, d 值的組合
    for a, b, c, d in itertools.product(range(a1, a2 + 1), range(b1, b2 + 1), range(c1, c2 + 1), range(d1, d2 + 1)):
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
                # M*N/2 => 3*3不符合 ?
                if period > M*N/2:
                    period = 0
                    break
                
            num += 1
            transformations.append([num, a, b, c, d, period])
            print(f"{num}, {a}, {b}, {c}, {d}, {period}")

    return transformations


def save_to_csv(M, N, a1, a2, b1, b2, c1, c2, d1, d2, data):
    # 將數據保存到CSV文件中
    filename = f"{M}_{N}_parameters.csv"
    
    # 寫入 CSV 檔案
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 寫入欄位名稱和數據
        writer.writerow(['M', 'N', 'a1', 'a2', 'b1', 'b2', 'c1', 'c2', 'd1', 'd2' ])
        writer.writerow([M, N, a1, a2, b1, b2, c1, c2, d1, d2])
        writer.writerow(['No.', 'a', 'b', 'c', 'd', 'period'])
        writer.writerows(data)

    print(f"Data has been written to {filename}")

def main():
    try:
        # 讀取影像解析度和轉換矩陣範圍
        M = int(input("M: "))
        N = int(input("N: "))

        print("Matrix:")
        a1 = int(input("a1: "))
        a2 = int(input("a2: "))
        b1 = int(input("b1: "))
        b2 = int(input("b2: "))
        c1 = int(input("c1: "))
        c2 = int(input("c2: "))
        d1 = int(input("d1: "))
        d2 = int(input("d2: "))

        # # 驗證 M 和 N 的比例
        # if not is_valid_ratio(M, N):
        #     print("M/N is not an integer or N/M is not an odd number")
        #     return

        # 生成轉換矩陣
        transformations = generate_transformations(M, N, a1, a2, b1, b2, c1, c2, d1, d2)
        
        # 並保存到 CSV
        if transformations:
            save_to_csv(M, N, a1, a2, b1, b2, c1, c2, d1, d2, transformations)
        else:
            print("No valid transformations found.")

    except ValueError as e:
        print(f"ValueError {str(e)}")
    except Exception as e:
        print(f"Exception: {str(e)}")

# def main(M, N, a1, a2, b1, b2, c1, c2, d1, d2):
#     # if not is_valid_ratio(M, N):
#     #     print("Error: M/N is not an integer and N/M is not an odd number.")
#     #     return
    
#     transformations = generate_transformations(M, N, a1, a2, b1, b2, c1, c2, d1, d2)
    
#     if transformations:
#         save_to_csv(M, N, transformations)
#     else:
#         print("No valid transformations found.")


# main(5, 5, 1, 3, 1, 3, 1, 3, 1, 3)
main()

