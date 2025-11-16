import os
import subprocess

import cv2
import numpy as np
from sympy.simplify.radsimp import collect_sqrt


def RawReader_planar(FileName, ImgWidth, ImgHeight, NumFramesToBeComputed):
    f = open(FileName, 'rb')
    frames = NumFramesToBeComputed
    width = ImgWidth
    height = ImgHeight
    data = f.read()
    f.close()
    data = [int(x) for x in data]

    data_list = []
    n = width * height
    for i in range(0, len(data), n):
        b = data[i:i + n]
        data_list.append(b)
    x = data_list

    listR = []
    listG = []
    listB = []
    for k in range(0, frames):
        R = np.array(x[3 * k]).reshape((width, height)).astype(np.uint8)
        G = np.array(x[3 * k + 1]).reshape((width, height)).astype(np.uint8)
        B = np.array(x[3 * k + 2]).reshape((width, height)).astype(np.uint8)
        listR.append(R)
        listG.append(G)
        listB.append(B)
    return listR, listG, listB
# 输入和输出文件夹路
Model = ['SCFR']
for model in Model:
    input_folder = "experiment/" + model + "/Iframe_YUV420/dec"  # 替换为您的 RGB 文件夹路径
    output_folder = "experiment/" + model + "/Iframe_YUV420/dec/yuv420"  # 替换为您的输出文件夹路径

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".rgb"):
            input_path = os.path.join(input_folder, filename)
            output_filename = filename.replace(".rgb", ".yuv")
            output_path = os.path.join(output_folder, output_filename)

            # 根据文件名设置帧率
            # if filename.startswith("CFVQA"):
            #     framerate = 125
            # elif filename.startswith("VOXCELEB"):
            #     framerate = 250
            # else:
            #     print(f"Unknown file type for {filename}. Skipping...")
            #     continue
            framerate = 125
            listR, listG, listB = RawReader_planar(input_path, 256, 256, framerate)
            f_temp = open(output_path,'w')
            for frame_idx in range(0, framerate):
                img_input_rgb = cv2.merge([listR[frame_idx], listG[frame_idx], listB[frame_idx]])
                img_input_yuv = cv2.cvtColor(img_input_rgb, cv2.COLOR_RGB2YUV_I420)  # COLOR_RGB2YUV
                img_input_yuv.tofile(f_temp)
            f_temp.close()

            print(f"Converted {input_path} to {output_path} with framerate {framerate}")

print("All conversions completed!")
