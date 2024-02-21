# RenderingTechniques
	Assignment for Rendering Techniques Course

Assign01-Image Encryption by 2D EAT and RP:
將使用者輸入之 EAT 轉換矩陣套用在圖像上，使各圖片逐像素進行 G 次之 EAT
轉換，再把各像素值依 RP 演算法進行調整。
以及使用 RRP 演算法、逆 EAT 矩陣來還原圖像。

Assign02-Image Encryption Using Enhanced Two-dimensional Sine Logistic 
Modulation (2D-SLM):
將圖片完成作業一的處理後，再進行 pixel diffusion 以及 Pixel scrambling 
using exclusive OR，並記錄 secret keys 參數。
最後再解密以及量測 MSE 是否等於 0。

Assign03-Determine the Rectangular Transformation Matrix:
利用輸入值計算符合條件的 legal transformation Matrix 係數，再計算這些
legal matrix 的 period 並保存結果。

Assign04-Determine the Period from the Matrix Coefficients:
利用數個參數計算執行 Rectangular Transformation 所需要的 period，並輸
出 M、N、矩陣(a, b, c, d)、period 等參數。

Assign05-Inverse Rectangular Transformation:
給定 a, b, c, d, M, N 等參數，使用 RT 演算法來加密影像，再將這些影像用
IRT 演算法解密，並記錄 round 及加解密執行時間。

Assign06-General Weighted Modulus Data Hiding:
讀取 PA table 參數檔，將這些參數應用 GWM 演算法把(掩護)影像嵌入 6、35、
87 進制秘密訊息，產生(偽裝)影像。

Assign07- General Weighted Modulus Reversible Data Hiding Algorithm:
以 GWMRDH 演算法對單張原始灰階影像嵌入秘密訊息，產生三張嵌密灰階影像，
再以相同演算法對三張嵌密灰階影像提取訊息，恢復原始單張灰階影像。
其中需防止 pixel overflow/underlow，最後計算 MSE 確認差異程度。

Assign08-Integrated Message Embedding and Encryption Algorithm:
先用 GWMRDH 嵌入訊息，再執行 Channel Composition、Permutation 最後再進
行 RT 產生彩色加密圖片。利用 IRT 將加密圖像解密，執行 Channel Inverse 
Permutation、Decomposition，並提取秘密訊息，最後再將圖片還原。
