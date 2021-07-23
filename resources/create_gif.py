import imageio
from pathlib import Path


with imageio.get_writer('pres.gif', mode='I', fps=15) as writer:
    for filename in sorted(list(Path("gif_images").iterdir())):
        image = imageio.imread(filename)
        writer.append_data(image)
