# pip install easyocr or pip install -r requirements.txt
import torch
import easyocr
import re


def text_recognition(file_path):
    reader = easyocr.Reader(["ru", "en"])
    result = reader.readtext(file_path, detail=0, paragraph=True)
    nums = []
    print(result)
    for line in result:
        numa = re.sub(r',', r'.', line)
        num = re.search(r"\b[-+]?(?:\d*\.*\d+)\b", numa)
        print(result)
        try:

            numa = float(re.sub(r',', r'.', num.group(0)))
            nums.append(numa)
        except:
            continue
    if min(nums) > 0:
        nums.insert(0, 0)
    print(nums)

    return max(nums), min(nums)


def main():
    max, min = text_recognition("Pressure Gauge/3.jpg")
    print(min, max)


if __name__ == "__main__":
    main()
