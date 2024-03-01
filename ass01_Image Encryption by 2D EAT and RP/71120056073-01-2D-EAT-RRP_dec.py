import cv2
import numpy as np
import random
import os
import configparser


#參數預設值
class Image_ReverseEAT:
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
## EAT逆轉換
def EAT_ReverseTransform(x, y, imageObj):
    '''
    EAT逆轉換矩陣
    | ab+1 -a |
    | -b    1 |
    x,y: (x,y)座標
    return: 轉換後座標(x', y')
    '''
    a = imageObj.a
    b = imageObj.b
    N = imageObj.N
    x_prime = int((a*b*x + x -a*y) % N)
    y_prime = int((-b*x + y) % N)
    
    return x_prime, y_prime




## 執行多次Reverse EAT
def EAT_reverse(imageObj):
    
    N = imageObj.N
    G = imageObj.G
    image = imageObj.image
    
    ### 確認圖片色彩
    if len(image.shape) == 2:
        new_image = np.zeros((N, N), dtype=np.uint8) 
    elif len(image.shape) == 3:
        new_image = np.zeros((N, N, 3), dtype=np.uint8)

    for time in range(G):
        print('G = ' + str(time)) 
        for x in range(N):
            for y in range(N):
                new_x, new_y = EAT_ReverseTransform(x, y, imageObj)
                new_image[new_x][new_y] = image[x][y]

    return new_image

def generate_permutation(random_integers):
    permutation = random_integers.copy()
    n = len(permutation)
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        permutation[i], permutation[j] = permutation[j], permutation[i]
    return permutation



def reverse_RP(encrypted_value, seed):
    random.seed(seed)
    random_integers = [i for i in range(8)]
    random.shuffle(random_integers)

    permutation = generate_permutation(random_integers)
    decrypted_value = 0
    for i in range(8):
        bit = (encrypted_value >> permutation[i]) & 1
        decrypted_value |= (bit << i)
    return decrypted_value



def reverse_imagePixel(input_path, output_path, seed):
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    decrypted_image = np.zeros_like(image)
    
    if len(image.shape) == 2:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                decrypted_image[i, j] = reverse_RP(image[i, j], seed)
    elif len(image.shape) == 3:
       for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    decrypted_image[i, j, k] = reverse_RP(image[i, j, k], seed)

    cv2.imwrite(output_path, decrypted_image)


#主程式
## 讀取 config.ini 文件     
config = configparser.ConfigParser()
configPath = os.path.join(os.path.dirname(__file__), 'config.ini')
config.read(configPath)

## 取得各項的值
a = config.getint('Parameter', 'a')
b = config.getint('Parameter', 'b')
G = config.getint('Parameter', 'G')
seed = config.getint('Parameter', 'seed')

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
        
        ### RRP
        reverse_imagePixel(enc_path, dec_path, seed)
        
        ### REAT
        image = cv2.imread(dec_path, cv2.IMREAD_UNCHANGED)
        N = image.shape[0]
        imageObj = Image_ReverseEAT(image, a, b, N, G)
        origin_image = EAT_reverse(imageObj)
        cv2.imwrite(dec_path, origin_image)
       

print('Decrypt Finished')






