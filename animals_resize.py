import glob
import tqdm

from PIL import Image
import os, sys

image_paths = glob.glob("data/animals/**/*")

for item in tqdm.tqdm(image_paths):
    im = Image.open(item)
    dir, filename = os.path.split(item)
    imResize = im.resize((300, 200), Image.ANTIALIAS)
    # print(dir.split('/'))
    # print("data/animals_resized/" + dir.split('/')[2] + "/" + filename)
    imResize.save("data/animals_resized/" + dir.split('/')[2] + "/" + filename, 'JPEG', quality=90)
    # exit(1)