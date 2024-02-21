import cv2
import numpy as np
import random
import os
import configparser
from math import*
import hashlib


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

def Pixel_Descrambling(img, R, P_0, C_0):
    dec_img = np.zeros_like(img)

    dec_img[0][2] = R[0] ^ img[0][2] ^ P_0[2] ^ C_0
    dec_img[0][1] = R[1] ^ img[0][1] ^ P_0[1] ^ img[0][2]
    dec_img[0][0] = R[2] ^ img[0][0] ^ P_0[0] ^ img[0][1]
    for i in range(1, img.shape[0]):
        dec_img[i][2] = R[i*3] ^ img[i][2] ^ dec_img[i-1][2] ^ img[i-1][0]
        dec_img[i][1] = R[i*3+1] ^ img[i][1] ^ dec_img[i-1][1] ^ img[i][2]
        dec_img[i][0] = R[i*3+2] ^ img[i][0] ^ dec_img[i-1][0] ^ img[i][1]
    return dec_img

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

def mean_squared_error(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    return err



#主程式
## 讀取 config.ini 文件     
config = configparser.ConfigParser()
configPath = os.path.join(os.path.dirname(__file__), 'config.ini')
config.read(configPath)

## 取得各項的值
eat_a = config.getint('Parameter', 'eat_a')
eat_b = config.getint('Parameter', 'eat_b')
G = config.getint('Parameter', 'G')
a_tilde = config.getint('Parameter', 'a_tilde')
b_tilde = config.getint('Parameter', 'b_tilde')
x_0 = config.getfloat('Parameter', 'x_0')
y_0 = config.getfloat('Parameter', 'y_0')
effect_g = config.getint('Parameter', 'effect_g')
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
        
        ### SHA512
        ori_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        hashed_img = hashlib.sha512(bytes(ori_image)).hexdigest()
        hashed_img = bin(int(hashed_img, base=16))

        ### Split hash number
        k = split_hash_number(hashed_img)
        
        ### Descrambling    
        a, b, x, y = diffusion_init(a_tilde, b_tilde, x_0, y_0, k)
        R = []
        for i in range(512*512*2):
            x, y = pixel_diffusion(a, b, x, y)
            R.append(int((x*(10**7))%256))
            R.append(int((y*(10**7))%256))
        P_0 = [k[4], k[5], k[6]]
        C_0 = k[7]
        image = cv2.imread(enc_path, cv2.IMREAD_UNCHANGED)
        shape = image.shape
        image = image.reshape(image.shape[0]*image.shape[1], 3)
        dec_img = Pixel_Descrambling(image, R, P_0, C_0)
        dec_img = dec_img.reshape(shape)
        cv2.imwrite(dec_path, dec_img)
        print('Descrambling Finished')

        ### RRP
        reverse_imagePixel(dec_path, dec_path, seed)
        print('RRP Finished')
        
        ### REAT
        RRP_image = cv2.imread(dec_path, cv2.IMREAD_UNCHANGED)
        N = shape[0]
        imageObj = Image_ReverseEAT(RRP_image, eat_a, eat_b, N, G)
        dec_image = EAT_reverse(imageObj)

        cv2.imwrite(dec_path, dec_image)
        
        ## MSE 
        origin_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        dec_image = cv2.imread(dec_path, cv2.IMREAD_UNCHANGED)
        if len(origin_image.shape) == 2:
            dec_image = cv2.cvtColor(dec_image, cv2.COLOR_BGR2GRAY)
        mse = mean_squared_error(ori_image, dec_image)
        print('mse = ' + str(mse)+', Dencrypt Finished')
       

print('Decrypt Finished')






