import string
import pandas as pd
import numpy as np
import PIL
import sys
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import math

class ncVisualize():
    """
        该对象针对netCDF格式数据绘图任务，
        值得注意的是，当传入数据包括有T、A数据，
        本模块将调用已有的机器学习模型补充演算结果。
        模块调用方法实例如下：
        from 文件位置.nccDrawer import ncVisualize
        a = ncVisualize
        a.address = "netCDF文件路径"
        a.scale = "linear or log"
        a.variables = "需要绘制的变量名"
        # a.shapes = [width height]
        # a.projection = "投影名称"
        a.show()
    """

    def __init__(self , address: string , scale: string , *args, **kwargs):
        super(self).__init__(*args , **kwargs)
        
        self.address = address
        self.scale = scale

        self.image = self.nc2np()
        self.linearORlog()
        self.coasetline()
        self.mask()
        self.tORa()
        self.output()

    def nc2np(self):
        ncDataset = nc.Dataset(self.address , "r")
        variables = ncDataset.variableskeys()
        print(variables)
        print("type index of varivables")
        index = input()
        target = ncDataset.variables[variables(index)]
        return np.array(target)
	
    def linearORlog(self):
        if self.scale == "linear":
            self.image = self.image
        elif self.scale == "log":
            self.image = math.log(self.image)
        else:
            print("please choose scale as linear or log")

    def coastline(self):
        fig=plt.figure(figsize=(20,15))#设置一个画板，将其返还给fig
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.coastlines()
        plt.show()

    # def mask(self):

    def tORa(self):
        if self.types == "T" or self.types == "A":
            a = 1

    # def ouput(self):