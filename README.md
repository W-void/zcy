# read me
  
1. 下载数据 ： wget -i ./dataLoad/result.txt -P ~/BC  

2. 解压 ： ls  ~/BC/*.tar.gz | xargs -n1 tar xzvf

3. 大图切割 ： python3 cropImage.py

4. 软连接训练集到工作目录 ： ln -s ～/BC/BC/train ${root}/VOC2012/train

5. 软连接测试集到工作目录 ：ln -s ～/BC/BC/val ${root}/VOC2012/val

6. 进行训练 ： python3 landsat.py