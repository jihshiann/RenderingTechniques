import random
import numpy as np
import cv2
import os
import configparser


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
def change_imagePixel(image, output_path, seed):
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

    cv2.imwrite(output_path, encrypted_image)

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
a = int(input("請輸入a的值:"))
b = int(input("請輸入b的值:"))

## 固定seed
seed = np.random.randint(0, 999)
## EAT執行次數
G = np.random.randint(5, 301)
        

## 寫入執行加密使用參數
setConfig(f'Parameter', 'a', a)
setConfig(f'Parameter', 'b', b)
setConfig(f'Parameter', 'G', G)
setConfig(f'Parameter', 'seed', seed)

## 讀取影像檔案
current_directory = os.getcwd()
source_folder = os.path.join(current_directory, 'source')
decryp_folder = os.path.join(current_directory, 'decryp')
encryp_folder = os.path.join(current_directory, 'encryp')

for filename in os.listdir(source_folder):
    if filename.endswith(('.png')):
        ### 檔名處理
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
        imageObj = Image_EAT(image, a, b, N, G)
        
        ### EAT
        EAT_image = EAT_exec(imageObj)
        
        ### RP
        enc_path = os.path.join(encryp_folder, enc_path)
        change_imagePixel(EAT_image, enc_path, seed)

print('Encrypt Finished')


