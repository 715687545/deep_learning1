import requests
from urllib.request import urlretrieve

import io

img_path="./photo/test.jpg"#input("请输入图片路径")
img = img_path # 图片路径
files = {"pic_path": open(img, "rb")}  # files # 类似data数据
url = "http://pic.sogou.com/pic/upload_pic.jsp"  # post的url
keywords = requests.post(url, files=files).text  # requests 提交图片
url2 = "http://pic.sogou.com/pic/ocr/ocrOnline.jsp?query=" + keywords  # keywords就是图片url此方式为get请求
ocrResult = requests.get(url2).json()  # 直接转换为json格式
    
contents = ocrResult['result']  # 类似字典 把result的value值取出来 是一个list然后里面很多json就是识别的文字
text = ""
for content in contents:  # 遍历所有结果
    text+=(content['content'].strip()+'\n')  # strip去除空格 他返回的结果自带一个换行
print(text)
