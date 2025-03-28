import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

with rasterio.open('your/file/path/os_terrain.tif') as dem:
    z = dem.read(1)  # reading the first band


#defining the extent
nrows, ncols = z.shape
x = np.linspace(dem.bounds.left, dem.bounds.right, ncols)
y = np.linspace(dem.bounds.top, dem.bounds.bottom, nrows)
x, y = np.meshgrid(x, y)

# Set up plot
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
# Reduce the number of ticks on the axes (Can look a bit cluttered otherwise!)
ax.xaxis.set_major_locator(MaxNLocator(1))
ax.yaxis.set_major_locator(MaxNLocator(1))
ax.zaxis.set_major_locator(MaxNLocator(4))

ls = LightSource(270, 45)

# Create a colormap that is a gradient between your two colors
cmap = LinearSegmentedColormap.from_list("mycmap", ["#fff3b2", "#f28522"])
# Currently this uses white to OS brand colour Orange, but you can configure any gradient you like


# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
rgb = ls.shade(z, cmap=cmap, vert_exag=0.5, blend_mode='overlay')

surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)
# Editing elevation and azimuth angle
ax.view_init(elev=50, azim=270)

# display your plot!
plt.show()