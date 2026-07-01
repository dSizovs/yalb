import glob, re, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

def fnum(f): return int(re.search(r'(\d+)', f).group(1))
frames = sorted(glob.glob("cavframe_*.txt"), key=fnum)
print("found", len(frames), "frames")

def load(fname):
    with open(fname) as fh:
        Nx, Ny = map(int, fh.readline().split())
        data = np.loadtxt(fh)
    ux = data[:,1].reshape(Ny, Nx)
    uy = data[:,2].reshape(Ny, Nx)
    speed = np.sqrt(ux**2 + uy**2)
    return Nx, Ny, ux, uy, speed

_,_,_,_,last_speed = load(frames[-1])
vmax = last_speed.max()

images=[]
for k,fname in enumerate(frames):
    Nx,Ny,ux,uy,speed = load(fname)
    fig, ax = plt.subplots(figsize=(4.6,4.2))
    ax.imshow(speed, origin="lower", cmap="viridis", vmin=0, vmax=vmax, extent=[0,1,0,1])
    xs = np.linspace(0,1,Nx); ys = np.linspace(0,1,Ny)
    X,Y = np.meshgrid(xs,ys)
    try:
        ax.streamplot(X, Y, ux, uy, color="white", density=0.7, linewidth=0.5, arrowsize=0.6)
    except Exception as e:
        print("streamplot skipped:", e)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Lid-driven cavity on GPU (A100)\nstep %d" % (k*1500), fontsize=10)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    images.append(img[:,:,:3].copy())
    plt.close(fig)

durations = [0.15]*len(images)
durations[-1] = 1.2
imageio.mimsave("cavity.gif", images, duration=durations, loop=0)
print("Saved cavity.gif with", len(images), "frames")
