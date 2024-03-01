1. 使用的函式庫：random, os, numpy, cv2, configparser, math, hashlib

2. 加密: EAT+RR+Scrambling；解密: RRP+REAT+DeScrambling

3. 執行:
	3-1. 執行71120056073-02-2D-EAT-RP-Chao_enc.py
		進行影像加密，並將結果及加密參數存入encryp資料夾

	3-2. 執行71120056073-01-2D-EAT-RRP_dec.py
		從encryp存取加密圖片以及參數進行解密，並將解密後圖片存入decryp資料夾
		最後再將解密圖片與原圖做MSE，確認還原之圖像與原圖一致