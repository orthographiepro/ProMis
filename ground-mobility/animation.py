import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation
import pandas as pd
from numpy import repeat, column_stack, unique, sum, eye, arange
from promis.geo import CartesianDeltaCollection, CartesianMap, DeltaGrid
from promis import StaRMap

name = "maxspeed_discrete"
fname = f"data/{name}"

## init
uam = CartesianMap.load(f"{fname}/{name}_uam.pkl")
star = StaRMap.load(f"{fname}/{name}_star.pkl")
support = CartesianDeltaCollection.load(f"{fname}/{name}_landscape.pkl")
params = star.get("maxspeed", "secondary").parameters

speeds = support.data["speed"].unique()
bearings = support.data["bearing"].unique()
print(speeds ,bearings)

fig, axes = plt.subplots(1,2, figsize = (6, 4))
img = params.scatter(plot_basemap=True, bearing=0, speed=30, alpha=0.8, ax=axes[0])
cbar = plt.colorbar(img, ax=axes[0], aspect=30, pad=0.02, shrink=0.8)
cbar.solids.set(alpha=0.8)
img = params.scatter(value_index=1, plot_basemap=True, bearing=0, speed=30, alpha=0.8, ax=axes[1])
cbar = plt.colorbar(img, ax=axes[1], aspect=30, pad=0.02, shrink=0.8)
cbar.solids.set(alpha=0.8)
fig.suptitle("probabilistic maxspeed")
axes[0].title.set_text("value")
axes[1].title.set_text("probability")
plt.show()

fig, axes = plt.subplots(2,2, figsize = (8,7))
fig.suptitle(f"varying speeds with prob. maxspeed")
for i, s in enumerate(speeds):
    ax = axes[i // 2, i % 2]
    ax.title.set_text(f"speed {s}")
    ax.title.set_fontsize(9)

    support.scatter(plot_basemap=True, bearing=0, speed=s, 
                    cmap="coolwarm_r", alpha=0.3, s=20, ax=ax)
    
plt.show()
0/0

target = DeltaGrid(uam.origin, (60, 60), 40, 40)
support.into(target)

s=30
b=0

frames = 36
ims = []

crossings: CartesianMap = uam.filter("crossing")
fig, ax = plt.subplots()
plt.title(f"variable bearings, speed {s}")
for i in range(frames):
    b = 360 / (frames) * i
    img = target.scatter(plot_basemap=True, bearing=b, speed=s, cmap="coolwarm_r", ax=ax, alpha=0.5, animated=True, method="slinear")
    [f.plot(ax, color="purple", alpha=1, animated=True) for f in crossings.features]

    if i == 0:
        target.scatter(plot_basemap=True, bearing=b, speed=s, cmap="coolwarm_r", ax=ax, alpha=0.5)
        [f.plot(ax, color="purple", alpha=1) for f in crossings.features]

    ims.append([img])

# img = target.scatter(plot_basemap=True, bearing=b, speed=s, cmap="coolwarm_r", ax=ax, alpha=0)
# [f.plot(ax, color="purple", alpha=1) for f in crossings.features]
# cbar = plt.colorbar(img, aspect=30, pad=0.02)
# cbar.solids.set(alpha=0.8)

# interpolate = support.get_interpolator()
# cmap = plt.get_cmap("coolwarm_r")
# unique_locations = target.unique_coordinates()

# def animate(frame):
#     b = 360 / (frames-1) * frame
#     coordinates = column_stack((
#         unique_locations[:, 0],
#         unique_locations[:, 1],
#         repeat(b, unique_locations.shape[0]),
#         repeat(s, unique_locations.shape[0]),
#     ))
#     img.set_alpha(0.6)
#     img.set_offsets(coordinates[:, :2])
#     img.set_color(cmap(interpolate(coordinates)))
#     return (img,)

# ani = FuncAnimation(fig=fig, func=animate, frames=frames, interval=1000, blit=True)
ani = ArtistAnimation(fig, ims, interval = 200, blit = True, )
ani.save(filename="data/bearing_interpolation_slinear.gif", writer="pillow")
plt.show()

    
