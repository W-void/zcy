# read me
  
1. 下载数据 ： wget -i ./dataLoad/result.txt -P ~/BC  

2. 解压 ： ls  ~/BC/*.tar.gz | xargs -n1 tar xzvf

3. 大图切割 ： python3 cropImage.py

4. 软连接训练集到工作目录 ： ln -s ～/BC/BC/train ~/Documents/landsat/VOC2012

5. 软连接测试集到工作目录 ：ln -s ～/BC/BC/val ~/Documents/landsat/VOC2012

6. 进行训练 ： python3 landsat.py