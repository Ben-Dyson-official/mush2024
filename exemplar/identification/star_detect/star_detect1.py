from PIL import Image;
import cv2;

star_image = Image.open("star_hubble.jpg");
pixels = star_image.load();
print star_image.size
print(pixels[22,22]);

star_image.show()

c = raw_input();
