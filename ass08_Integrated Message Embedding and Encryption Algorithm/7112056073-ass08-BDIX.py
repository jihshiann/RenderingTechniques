import re
import os
import cv2
import csv
import copy
import math
import numpy as np
import pandas as pd


def write_csv(csv_name, n, M, W, MSE, PSNR, EC, ER):
    with open(f'15-imgres/{csv_name}', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['RPA', str(n), str(M), 'w1', 'w2', 'w3'])
        writer.writerow(['Index', 'd', 'SE', str(W[0]), str(W[1]), str(W[2])])
        writer.writerow(['MSE', str(MSE)])
        writer.writerow(['PSNR', str(PSNR)])
        writer.writerow(['EC', str(EC)])
        writer.writerow(['ER', str(ER)])


def GMWRDH_Restoration(image1, image2, image3, n):
    image_origin = copy.deepcopy(image1)
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            P_prime = np.array([image1[i][j], image2[i][j], image3[i][j]])
            P = round(np.sum(P_prime) / n)
            image_origin[i][j] = P
    return image_origin
def GMWRDH_Message_Extraction(image1, image2, image3, W, M, seed):
    secret_messages = []
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            P_prime = np.array([image1[i][j], image2[i][j], image3[i][j]])
            S = np.dot(P_prime, W.T) % M
            secret_messages.append(S)
    with open(f"14-mesext/mes_ext_{int(seed/100)}.txt", "w") as mesext_file:
        for secret_message in secret_messages:
            mesext_file.write("%s " % secret_message)
    return secret_messages


def channel_decomposition(image):
    image_change1 = image[:, :, 2]
    image_change2 = image[:, :, 1]
    image_change3 = image[:, :, 0]
    return image_change1, image_change2, image_change3
def channel_inverse_permutation(image, PK):
    image_change = copy.deepcopy(image)
    match PK:
        case 1:
            image_change[:, :, 0] = image[:, :, 0]
            image_change[:, :, 1] = image[:, :, 2]
            image_change[:, :, 2] = image[:, :, 1]
        case 2:
            image_change[:, :, 0] = image[:, :, 1]
            image_change[:, :, 1] = image[:, :, 2]
            image_change[:, :, 2] = image[:, :, 0]
        case 3:
            image_change[:, :, 0] = image[:, :, 1]
            image_change[:, :, 1] = image[:, :, 0]
            image_change[:, :, 2] = image[:, :, 2]
        case 4:
            image_change[:, :, 0] = image[:, :, 2]
            image_change[:, :, 1] = image[:, :, 0]
            image_change[:, :, 2] = image[:, :, 1]
        case 5:
            image_change[:, :, 0] = image[:, :, 2]
            image_change[:, :, 1] = image[:, :, 1]
            image_change[:, :, 2] = image[:, :, 0]
    return image_change


def read_RPA_table(RPA_table_file_name):
    RPA_table = pd.read_csv(f'10-rpatab/{RPA_table_file_name}')
    RPA_table = RPA_table.drop(columns=[RPA_table.columns[0], RPA_table.columns[2], RPA_table.columns[6]], axis=1)
    RPA_table = RPA_table.iloc[:-3, :]
    columns_name = RPA_table.iloc[0].tolist()
    RPA_table.columns = columns_name
    RPA_table = RPA_table.drop(RPA_table.index[0])
    RPA_table = RPA_table.reset_index(drop=True).astype(int)
    RPA_table_name_numbers = re.findall(r'\d+', RPA_table_file_name)
    n = int(RPA_table_name_numbers[0])
    M = int(RPA_table_name_numbers[1])
    W = np.array(RPA_table_name_numbers[2:-1]).astype(int)
    Z = int(RPA_table_name_numbers[-1])
    return RPA_table, n, M, W, Z


def find_S(t, p):
    S = 1
    while (S * t - 1) % p != 0:
        S += 1
    return S
def inverse_transform_step1(x, y, a, b, c, d, M, N):
    t = a * d - b * c
    p = math.gcd(M, N)
    S = find_S(t, p)
    x_and_y_p = np.mod(np.dot((S * np.array([[d, -b], [-c, a]])), np.array([[x], [y]])), p)
    return x_and_y_p[0][0], x_and_y_p[1][0]

def inverse_transform_step2(x, y, x_p, y_p, a, b, c, d, M, N):
    p = math.gcd(M, N)
    h = M / p
    v = N / p
    H = ((x - (a * x_p) - (b * y_p)) / p) % h
    V = ((y - (c * x_p) - (d * y_p)) / p) % v
    return H, V

def inverse_transform_step3(H, V, a, b, c, d, M, N):
    p = math.gcd(M, N)
    h = M / p
    v = N / p
    x_h = (find_S(a, h) * H) % h
    y_v = (find_S(d, v) * V) % v
    return x_h, y_v

def inverse_transform_step4(x_p, y_p, x_h, y_v, M, N):
    p = math.gcd(M, N)
    x = x_p + p * x_h
    y = y_p + p * y_v
    return int(x), int(y)
def inverse_rectangular_transform(image, a, b, c, d, M, N, G):
    image_change = copy.deepcopy(image)
    for i in range(N):
        for j in range(M):
            x_p, y_p = inverse_transform_step1(j, i, a, b, c, d, M, N)
            H, V = inverse_transform_step2(j, i, x_p, y_p, a, b, c, d, M, N)
            x_h, y_v = inverse_transform_step3(H, V, a, b, c, d, M, N)
            change_x, change_y = inverse_transform_step4(x_p, y_p, x_h, y_v, M, N)
            image_change[change_y][change_x] = image[i][j]
    return image_change


def read_RT_secret_key(file_path):
    with open(file_path, 'r') as f:
        line = f.readline().split(' ')
        return [int(line[i]) for i in range(6)] + [int(f.readline())]



def calculate_MSE(image_new_name):
    image_origin_name = image_new_name.split('_')
    image_origin_name = image_origin_name[0] + '.png'
    image_origin = cv2.imread(f'1-origin/{image_origin_name}', cv2.IMREAD_UNCHANGED)
    image_new = cv2.imread(f'9-restor/{image_new_name}', cv2.IMREAD_UNCHANGED)
    MSE = np.mean((image_origin - image_new)**2)
    return MSE
def calculate_PSNR(MSE):
    if MSE == 0:
        return float('inf') 
    PSNR = round(10 * np.log10((255**2) / MSE), 2)
    return PSNR
def calculate_EC(H, V, M):
    EC = round(H * V * np.log2(M), 0)
    return EC
def calculate_ER(H, V, M):
    ER = round(H * V * np.log2(M) / 3, 5)
    return ER


#MAIN
folder_path = "5-encry"
image_files = os.listdir(folder_path)
RPA_table_files = os.listdir("10-rpatab")
np.random.seed(24)
PK = np.random.randint(0, 6, size=len(image_files))

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 讀取RT Secret Key
    a, b, c, d, M, N, G = read_RT_secret_key('13-decpar/%s-Secret-Key.txt' % image_file[:-8])

    # 進行逆矩形變換
    decryp_image = inverse_rectangular_transform(image, a, b, c, d, M, N, G)

    # 新圖像名稱
    image_new_name1 = image_file[:-8] + '_dec' + image_file[-4:]
    cv2.imwrite('6-decry/%s' % image_new_name1, decryp_image)

    # 讀取RPA table
    RPA_table, n, M, W, Z = read_RPA_table(RPA_table_files[image_files.index(image_file)])

    # 頻道逆排列
    channel_inverse_permutation_image = channel_inverse_permutation(decryp_image, PK[image_files.index(image_file)])

    # 保存逆排列後圖片
    image_new_name2 = image_file[:-8] + '_invmut_N' + str(n) + '_M' + str(M) + '_' + str(W[0]) + '_' + str(W[1]) + '_' + str(W[2]) + '_Z' + str(Z) + image_file[-4:]
    cv2.imwrite('7-invmut/%s' % image_new_name2, channel_inverse_permutation_image)

    # 圖片分解三通道
    image_change1, image_change2, image_change3 = channel_decomposition(channel_inverse_permutation_image)

    # 保存分解後圖片
    image_new_name3 = image_file[:-8] + '_decom_N' + str(n) + '_M' + str(M) + '_' + str(W[0]) + '_' + str(W[1]) + '_' + str(W[2]) + '_Z' + str(Z) + '_I1' + image_file[-4:]
    image_new_name4 = image_file[:-8] + '_decom_N' + str(n) + '_M' + str(M) + '_' + str(W[0]) + '_' + str(W[1]) + '_' + str(W[2]) + '_Z' + str(Z) + '_I2' + image_file[-4:]
    image_new_name5 = image_file[:-8] + '_decom_N' + str(n) + '_M' + str(M) + '_' + str(W[0]) + '_' + str(W[1]) + '_' + str(W[2]) + '_Z' + str(Z) + '_I3' + image_file[-4:]
    cv2.imwrite('8-decom/%s' % image_new_name3, image_change1)
    cv2.imwrite('8-decom/%s' % image_new_name4, image_change2)
    cv2.imwrite('8-decom/%s' % image_new_name5, image_change3)

    # 解密訊息
    secret_messages = GMWRDH_Message_Extraction(image_change1, image_change2, image_change3, W, M, (image_files.index(image_file) + 1) * 100)

    # 還原解密圖片
    image_origin = GMWRDH_Restoration(image_change1, image_change2, image_change3, n)

    # 保存還原後圖片
    image_new_name6 = image_file[:-8] + '_restor_N' + str(n) + '_M' + str(M) + '_' + str(W[0]) + '_' + str(W[1]) + '_' + str(W[2]) + '_Z' + str(Z) + image_file[-4:]
    cv2.imwrite('9-restor/%s' % image_new_name6, image_origin)

    # 計算誤差
    MSE = calculate_MSE(image_new_name6)
    PSNR = calculate_PSNR(MSE)
    EC = calculate_EC(image_origin.shape[1], image_origin.shape[0], M)
    ER = calculate_ER(image_origin.shape[1], image_origin.shape[0], M)

    # 紀錄嵌密與取密結果
    csv_name = image_file[:-8] + '_qualit_N' + str(n) + '_M' + str(M) + '_' + str(W[0]) + '_' + str(W[1]) + '_' + str(W[2]) + '_Z' + str(Z) + '.csv'
    write_csv(csv_name, n, M, W, MSE, PSNR, EC, ER)

