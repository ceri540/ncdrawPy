## this has been commit in github
from netCDF4 import Dataset as NetCDFFile 
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from pyproj import transform

## refer from https://www2.atmos.umd.edu/~cmartin/python/examples/netcdf_example1.html
# note this file is 2.5 degree, so low resolution data
nc = NetCDFFile(r"F:/data/2021daily/A2020001.L3m_DAY_RRS_Rrs_412_4km.nc")
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]
Rrs_412 = nc.variables['Rrs_412'][:]

# projection, lat/lon extents and resolution of polygons to draw
# resolutions: c - crude, l - low, i - intermediate, h - high, f - full
map = Basemap(projection='merc',llcrnrlon=-93.,llcrnrlat=35.
    ,urcrnrlon=-73.,urcrnrlat=45.,resolution='i') 

map.drawcoastlines()
map.drawstates()
map.drawcountries()
# can use HTML names or codes for colors
map.drawlsmask(land_color='Linen', ocean_color='#CCFFFF') 
# you can even add counties (and other shapefiles!)
map.drawcounties()

# make latitude lines ever 5 degrees from 30N-50N
parallels = np.arange(30,50,5.) 
# make longitude lines every 5 degrees from 95W to 70W
meridians = np.arange(-95,-70,5.) 
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)

lons,lats= np.meshgrid(lon-180,lat) 
# for this dataset, longitude is 0 through 360, 
# so you need to subtract 180 to properly display on map
x,y = map(lons,lats)

clevs = np.arange(960,1040,4)
cs = map.contour(x,y,Rrs_412[0,:,:]/100.,clevs,colors='blue',linewidths=1.)


# contour labels
plt.clabel(cs, fontsize=9, inline=1)
plt.title('Mean Sea Level Pressure')
plt.show()
plt.savefig('2m_temp.png')

## refer from https://stackoverflow.com/questions/65377866/working-with-netcdf-on-python-with-matplotlib
import imp
import xarray as xr
import matplotlib.pyplot as plt
dataset = xr.open_dataset(r"D:\ceeres\03_Program\01_SYSUM\ncDrawer\T2022071025500_L2_polymer_sub.nc")
ds = xr.open_dataset(r"F:\data\2021daily\A2020001.L3m_DAY_RRS_Rrs_412_4km.nc")
ds
ds['Rrs_412'].mean(['lon','lat']).plot()
plt.show()
ds['u10'].isel(latitude=10,longitude=5).plot()
ds['u10'].sel(latitude=-15,longitude=40,method='nearest').plot()

## refer from https://joehamman.com/2013/10/12/plotting-netCDF-data-with-Python/
from netCDF4 import Dataset
import numpy as np
my_example_nc_file = r"F:\data\2021daily\A2020001.L3m_DAY_RRS_Rrs_412_4km.nc"
fh = Dataset(my_example_nc_file, mode='r')
lons = fh.variables['lon'][:]
lats = fh.variables['lat'][:]
tmax = fh.variables['Tmax'][:]

tmax_units = fh.variables['Tmax'].units
fh.close()

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Get some parameters for the Stereographic Projection
lon_0 = lons.mean()
lat_0 = lats.mean()

m = Basemap(width=5000000,height=3500000,
            resolution='l',projection='stere',\
            lat_ts=40,lat_0=lat_0,lon_0=lon_0)

# Because our lon and lat variables are 1D,
# use meshgrid to create 2D arrays
# Not necessary if coordinates are already in 2D arrays.
lon, lat = np.meshgrid(lons, lats)
xi, yi = m(lon, lat)

# Plot Data
cs = m.pcolor(xi,yi,np.squeeze(tmax))

# Add Grid Lines
m.drawparallels(np.arange(-80., 81., 10.), labels=[1,0,0,0], fontsize=10)
m.drawmeridians(np.arange(-180., 181., 10.), labels=[0,0,0,1], fontsize=10)

# Add Coastlines, States, and Country Boundaries
m.drawcoastlines()
m.drawstates()
m.drawcountries()

# Add Colorbar
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label(tmax_units)

# Add Title
plt.title('DJF Maximum Temperature')

plt.show()

## refer from https://blog.csdn.net/theonegis/article/details/50805408
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

meteo_file = "/home/theone/Data/GreatKhingan/MERRA/MERRA2_400.inst1_2d_lfo_Nx.20131201.nc4"
fh = Dataset(meteo_file, mode='r')

# 获取每个变量的值
lons = fh.variables['lon'][:]
lats = fh.variables['lat'][:]
tlml = fh.variables['TLML'][:]

tlml_units = fh.variables['TLML'].units


# 经纬度平均值
lon_0 = lons.mean()
lat_0 = lats.mean()

m = Basemap(lat_0=lat_0, lon_0=lon_0)
lon, lat = np.meshgrid(lons, lats)
xi, yi = m(lon, lat)

# Plot Data
# 这里我的tlml数据是24小时的，我这里只绘制第1小时的（tlml_0）
tlml_0 = tlml[0:1:, ::, ::]
cs = m.pcolor(xi, yi, np.squeeze(tlml_0))

# Add Grid Lines
# 绘制经纬线
m.drawparallels(np.arange(-90., 91., 20.), labels=[1,0,0,0], fontsize=10)
m.drawmeridians(np.arange(-180., 181., 40.), labels=[0,0,0,1], fontsize=10)

# Add Coastlines, States, and Country Boundaries
m.drawcoastlines()
m.drawstates()
m.drawcountries()

# Add Colorbar
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label(tlml_units)

# Add Title
plt.title('Surface Air Temperature')
plt.show()

fh.close()

## refer from https://www.heywhale.com/mw/project/5f3c95a3af3980002cbf560b
import numpy as np
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER,LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader,natural_earth
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.image import imread

def createMap():
    #set shapfile directory
    shpPath = r""
    tifPath = r""
    proj = ccrs.PlateCarree()   # set projection
    fig = plt.figure(figsize = (15 , 25) , dpi = 600)
    axes = fig.subplots(1 , 1 , subplot_kw = {'projection' : proj})

    # map properties
    province = cfeat.ShapelyFeature(
        Reader(shpPath + '').geometries(),
        proj , edgecolor = 'k',
        facecolor = 'none'
    )

    # add feature
    axes.add_feature(province , linewidth = 0.6 , zorder = 2)
    # add coastline
    axes.add_feature(cfeat.COASTLINE.with_scale('50m') , linewidth = 0.6 , zorder = 10)
    # add river
    axes.add_feature(cfeat.RIVERS.with_scale('50m') , zorder = 10)
    axes.set_extent([105 , 133 , 15 , 45])
    # add high resolution height
    axes.imshow(
        imread(tifPath  + '') ,
        origin = 'upper' , transform = proj ,
        extent = [-180 , 180 , -90 , 90]
    )
    # set grid properties
    gridLine = axes.gridlines(
        crs = proj ,
        drawLabels = True ,
        linewidth = 0.6 ,
        color = 'k' ,
        alpha = 0.5 ,
        linestyle = '--'
    )
    gridLine.xlabels_top = False    # set off top grid labels
    gridLine.ylabels_right = False  # set off right grid labels
    gridLine.xformatter = LONGITUDE_FORMATTER   # set x grid as longitude
    gridLine.yformatter = LATITUDE_FORMATTER    # set y gird as latitude
    gridLine.xlocator = mticker.FixedLocator(np.arrange(95 , 145 + 5 , 5))
    gridLine.ylocator = mticker.FixedLocator(np.arrange(-5 , 45 + 5 , 5))

    # set sub graph
    left , bottom , width , height = 0.67 , 0.16 , 0.23 , 0.27
    axes2 = fig.add_axes(
        [left , bottom , width , height] ,
        projection = proj
    )
    axes2.add_feature(province, linewidth=0.6, zorder=2)
    axes2.add_feature(cfeat.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=10)  # 加载分辨率为50的海岸线
    axes2.add_feature(cfeat.RIVERS.with_scale('50m'), zorder=10)  # 加载分辨率为50的河流
    axes2.add_feature(cfeat.LAKES.with_scale('50m'), zorder=10)  # 加载分辨率为50的湖泊
    axes2.set_extent([105, 125, 0, 25])
    # add high resolution height
    axes2.imshow(
        imread(tifPath + ''),
        origin='upper',
        transform=proj,
        extent=[-180, 180, -90, 90]
    )
    return axes

if __name__ == '__main__':
    axes = createMap()
    plt.show()

    df = pd.read_csv(r'/home/kesci/work/buyo_position.csv')
    df

    # --调用刚才定义的地图函数
    ax = createMap()
    df['lon'] = df['lon'].astype(np.float64)
    df['lat'] = df['lat'].astype(np.float64)

    # --绘制散点图
    ax.scatter(
        df['lon'].values,
        df['lat'].values,
        marker='o',
        s=10 ,
        color ="blue"
    )

    # --添加浮标名称
    for i, j, k in list(zip(df['lon'].values, df['lat'].values, df['name'].values)):
        ax.text(i - 0.8, j + 0.2, k, fontsize=6)

    # --添加标题&设置字号
    title = f'distribution of station around China'
    ax.set_title(title, fontsize=18)
    plt.show()