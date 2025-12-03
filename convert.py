import pandas as pd
import json

def convert():
    # 读取 Excel 文件
    df = pd.read_excel('20251202-TKY.xlsx')
    
    # 将 DataFrame 转换为字典列表
    data = df.to_dict(orient='records')
    
    # 保存为 JSON 文件
    with open('input.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"转换完成，共 {len(data)} 条记录")

if __name__ == "__main__":
    convert()
