import os
import csv
import requests

filename = 'PDF_Image_to_test.csv'
out_dir = './img_inputs'

with open(filename, mode='r', encoding='utf-8') as file:
    # 创建 DictReader，自动用第一行作为表头
    csv_dict_reader = csv.DictReader(file)
    
    # 遍历每一行（每行是一个字典）
    for row in csv_dict_reader:
        print(row['task_id'], row['task_row_id'], row['task_data_image'])  # 通过表头访问字段
        resp = requests.get(row['task_data_image'])
        with open(os.path.join(out_dir, f"{row['task_id']}-{row['task_row_id']}.png"), 'wb') as f:
            f.write(resp.content)
        
        