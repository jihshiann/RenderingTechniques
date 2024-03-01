import random
import numpy as np
import cv2
import os
import configparser
import hashlib
from math import*


#參數預設值
class Image_EAT:
    '''
    image: numpy.ndarray
    a, b: 轉換矩陣參數
    N: 像素
    G: EAT轉換次數
    '''
    def __init__(self, image, a, b, N, G):
        self.image = image
        self.a, self.b = a, b
        self.N = N
        self.G = G
      

#方法
## EAT轉換
def EAT_transform(x, y, imageObj):
    '''
    EAT轉換矩陣
    | 1   a   |
    | b a*b+1 |
    x,y: (x,y)座標
    return: 轉換後座標(x', y')
    '''
    a = imageObj.a
    b = imageObj.b
    N = imageObj.N
    x_prime = int((x + a*y) % N)
    y_prime = int((b*x + (a*b+1)*y) % N)
    
    return x_prime, y_prime


## 計算EAT循環
def calculate_cycle(imageObj):
    '''
    計算EAT循環
    return: Cycle數
    '''
    N = imageObj.N
    x = np.random.randint(1, N)
    y = np.random.randint(1, N)
    ab_visited = set()
    current_x, current_y = x, y
    ab_cycle = 0
    consecutive_matches = 0
    
    while True:
        if (current_x, current_y) in ab_visited:
            consecutive_matches += 1
        else:
            consecutive_matches = 0
        
        ab_visited.add((current_x, current_y))
        current_x, current_y = EAT_transform(current_x, current_y, imageObj)
        ab_cycle += 1

        if consecutive_matches == 2:
            return ab_cycle-2


## 執行多次EAT
def EAT_exec(imageObj):
    """
     Args:
        image (numpy.ndarray): 原始影像，一個 2D 數字陣列
        a (int): 矩陣中的 a 值
        b (int): 矩陣中的 b 值
        G (int): EAT 轉換的次數
        N (int): 影像解析度
    Return:
        numpy.ndarray: 加密後的影像，與原始影像大小相同的 2D 數字陣列
    """
    N = imageObj.N
    G = imageObj.G
    image = imageObj.image
    eat_cycle = calculate_cycle(imageObj)
    
    ### 確認圖片色彩
    if len(image.shape) == 2:
        new_image = np.zeros((N, N), dtype=np.uint8) 
    elif len(image.shape) == 3:
        new_image = np.zeros((N, N, 3), dtype=np.uint8)
    else:
        print("?")
    
    while G % eat_cycle == 0:
        np.random.randint(5, 301)
    for time in range(G):
        print('G = ' + str(time)) 
        for x in range(N):
            for y in range(N):
                new_x, new_y = EAT_transform(x, y, imageObj)
                new_image[new_x][new_y] = image[x][y]
    return new_image


## 隨機排列
def generate_permutation(random_integers):
    permutation = random_integers.copy()
    n = len(permutation)
    ### Fisher-Yates
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        permutation[i], permutation[j] = permutation[j], permutation[i]
    return permutation


## RP
def random_permutation(pixel_value, seed):
    random.seed(seed)
    random_integers = [i for i in range(8)]
    random.shuffle(random_integers)

    permutation = generate_permutation(random_integers)
    encrypted_value = 0
    for i in range(8):
        ## 右位移i位元，將最右邊位元放至第permutation[i]位元
        bit = (pixel_value >> i) & 1
        encrypted_value |= (bit << permutation[i])
    return encrypted_value


## 更改圖片像素值
def change_imagePixel(image, seed):
    encrypted_image = np.zeros_like(image)

    if len(image.shape) == 2:
         for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                encrypted_image[i, j] = random_permutation(image[i, j], seed)
    elif len(image.shape) == 3:
         for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    encrypted_image[i, j, k] = random_permutation(image[i, j, k], seed)
    return encrypted_image

## 切割需要使用的hash值-new
def split_hash_number(hash_number):
    k = []
    k.append(int(hash_number[2 +effect_g:18+effect_g], 2))
    k.append(int(hash_number[18+effect_g:34+effect_g], 2))
    k.append(int(hash_number[34+effect_g:42+effect_g], 2))
    k.append(int(hash_number[42+effect_g:50+effect_g], 2))
    k.append(int(hash_number[50+effect_g:58+effect_g], 2))
    k.append(int(hash_number[58+effect_g:66+effect_g], 2))
    k.append(int(hash_number[66+effect_g:74+effect_g], 2))
    k.append(int(hash_number[74+effect_g:82+effect_g], 2))
    return k

## pixel diffusion參數初始化
def diffusion_init(a, b, x, y, k):
    a += k[0]/2**16
    b += k[1]/2**16
    x += k[2]/2**8
    y += k[3]/2**8

    return a, b, x, y

## pixel diffusion
def pixel_diffusion(a, b, x, y):
    x_new = sin(pi*(1-a*(x**2)+y))
    y_new = sin(pi*b*x)
    return x_new, y_new

## Pixel Scrambling
def Pixel_Scrambling(img, R, P_0, C_0):
    enc_img = np.zeros_like(img)
    
    enc_img[0][2] = R[0] ^ img[0][2] ^ P_0[2] ^ C_0
    enc_img[0][1] = R[1] ^ img[0][1] ^ P_0[1] ^ enc_img[0][2]
    enc_img[0][0] = R[2] ^ img[0][0] ^ P_0[0] ^ enc_img[0][1]
    for i in range(1, img.shape[0]):
        enc_img[i][2] = R[i*3] ^ img[i][2] ^ img[i-1][2] ^ enc_img[i-1][0]
        enc_img[i][1] = R[i*3+1] ^ img[i][1] ^ img[i-1][1] ^ enc_img[i][2]
        enc_img[i][0] = R[i*3+2] ^ img[i][0] ^ img[i-1][0] ^ enc_img[i][1]
    return enc_img

## config
def setConfig(section, attr, value):
    config.set(section, attr, str(value))
    with open(configPath, 'w') as config_file:
        config.write(config_file)
 
config = configparser.ConfigParser()
configPath = os.path.join(os.path.dirname(__file__), 'config.ini')
config.read(configPath)


#主程式
## 設定矩陣參數
eat_a = 2
eat_b = 5

## 固定seed
seed = 2023
## EAT執行次數
G = 8

## pixel diffusion 參數
a_tilde = 500
b_tilde = 500
x_0 = 0.1
y_0 = 0.1
effect_g = 9

## 寫入執行加密使用參數
setConfig(f'Parameter', 'eat_a', eat_a)
setConfig(f'Parameter', 'eat_b', eat_b)
setConfig(f'Parameter', 'G', G)
setConfig(f'Parameter', 'seed', seed)
setConfig(f'Parameter', 'a_tilde', a_tilde)
setConfig(f'Parameter', 'b_tilde', b_tilde)
setConfig(f'Parameter', 'x_0', x_0)
setConfig(f'Parameter', 'y_0', y_0)
setConfig(f'Parameter', 'effect_g', effect_g)


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

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        ### 取得影像的解析度
        N = image.shape[0]

        ### 創建EAT物件
        imageObj = Image_EAT(image, eat_a, eat_b, N, G)

        ### EAT
        EAT_image = EAT_exec(imageObj)
        print('EAT Finished')

        ### RP
        RP_image = change_imagePixel(EAT_image, seed)
        print('RP Finished')

        ### SHA512
        hashed_img = hashlib.sha512(bytes(image)).hexdigest()
        hashed_img = bin(int(hashed_img, base=16))

        ### Split hash number
        k = split_hash_number(hashed_img)
        
        ### diffusion
        a, b, x, y = diffusion_init(a_tilde, b_tilde, x_0, y_0, k)
        R = []
        for i in range(N*N*2):
            x, y = pixel_diffusion(a, b, x, y)
            R.append(int((x*(10**7))%256))
            R.append(int((y*(10**7))%256))   
            
        ### Scrambling
        P_0 = [k[4], k[5], k[6]]
        C_0 = k[7]
        shape = RP_image.shape
        if len(shape) == 2:
            RP_image = cv2.cvtColor(RP_image, cv2.COLOR_GRAY2BGR)
            shape = RP_image.shape

        RP_image = RP_image.reshape(RP_image.shape[0]*RP_image.shape[1], 3)        
        Scram_image = Pixel_Scrambling(RP_image, R, P_0, C_0)
        enc_img = Scram_image.reshape(shape)
        
        ### write img
        enc_path = os.path.join(encryp_folder, enc_path)
        cv2.imwrite(enc_path, enc_img)
        
        ### 密鑰記錄檔
        MatrixCoefficients = 'Matrix coefficients: (a, b): a = ' + str(eat_a) + ', b = ' +  str(eat_b)
        EncryptionRound =  'Encryption round: G = ' + str(G)
        RandomPermutationSeed = 'Random permutation seed: RS = ' + str(seed)
        ControlParameters =  'Control parameters (a, b): a = ' + str(a)+ ', b = ' +  str(b)
        InitialValues = 'Initial values (x0, y0): x0 = ' + str(x_0) + ', y0 = ' + str(y_0)
        TransientEffectConstant = 'Transient effect constant: g = ' + str(effect_g)
        VirtualHostPixels = 'Virtual host pixels (P0R, P0G, P0B): P0R = ' + str(P_0[0]) + ', P0G = ' + str(P_0[1]) + ', P0B = ' + str(P_0[2])
        VirtualEncryptedPixel = 'Virtual encrypted pixel: C0 = ' + str(C_0)
        text_lines = [MatrixCoefficients, EncryptionRound, RandomPermutationSeed, ControlParameters, InitialValues, TransientEffectConstant, VirtualHostPixels, VirtualEncryptedPixel]

        
        with open(encryp_folder + '/' + filename+'-Secret-Key.txt', 'w') as file:
            for line in text_lines:
                file.write(line + '\n')
                file.write('\n')
                
        print('Encrypt Finished')



