import re
import os

# 如果是在 .ipynb 环境下运行，这将会失败，然后使用当前工作目录
try:
    # 适用于 .py 文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # 适用于 .ipynb 笔记本
    current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

def hex_to_bin_with_spaces(hex_str):
    bin_str = bin(int(hex_str, 16))[2:].zfill(128)
    return ' '.join(bin_str[i:i+4] for i in range(0, len(bin_str), 4))

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    with open(output_file, 'w', encoding='utf-8') as file:
        i = 0
        while i < len(lines):
            line = lines[i]
            if "HMMA" not in line:
                i += 1
                continue  # 忽略不包含HMMA的行
            
            match1 = re.search(r'/\* (0x[0-9a-fA-F]+) \*/', line)
            if match1 and (i + 1) < len(lines):
                next_line = lines[i + 1]
                match2 = re.search(r'/\* (0x[0-9a-fA-F]+) \*/', next_line)
                if match2:
                    hex_num1 = match1.group(1)
                    hex_num2 = match2.group(1)
                    print("hex_num1:", hex_num1)
                    print("hex_num2:", hex_num2)
                    combined_hex = hex_num1 + hex_num2[2:]  # 拼接两个16进制数，去掉第二个数的'0x'
                    print("combined_hex:", combined_hex)
                    bin_num = hex_to_bin_with_spaces(combined_hex)
                    new_line = re.sub(r'/\* (0x[0-9a-fA-F]+) \*/', f'/* {bin_num} */', line.rstrip()) + '\n'
                    hmma_index = new_line.find("HMMA")
                    new_line = new_line[hmma_index:]
                    file.write(new_line)
                    i += 2
                    continue
            file.write(line)
            i += 1

file_name = '19-CuAssembler-test/asmFP16ref.sass'
input_file = os.path.join(current_dir, file_name)  # 替换为你的输入文件名
output_file = os.path.join(current_dir, file_name.replace('.sass', '.translated.sass'))  # 替换为你的输出文件名
process_file(input_file, output_file)
