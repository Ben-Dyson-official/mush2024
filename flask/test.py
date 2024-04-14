from tensorflow.keras.models import load_model
import cv2
import numpy as np 


def check_model(model_path, image_path):
    model = load_model(model_path)
    
    preped_image = prep_image(image_path, target_size=(224, 224))

    prediction = model.predict(preped_image)

    predicted_class = np.argmax(prediction, axis=1)

    return predicted_class

def prep_image(image_path, target_size):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("File not found")

    
    image = process_img(img)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 2)

	# threshold the image to reveal light regions in the
	# blurred image
    thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]

	# perform a series of erosions and dilations to remove
	# any small blobs of noise from the thresholded image
    thresh = cv2.erode(thresh, None, iterations=0)
    thresh = cv2.dilate(thresh, None, iterations=1)

    star_mask = get_star_mask(thresh)
	
    img = cv2.resize(star_mask, target_size)

    img = np.expand_dims(img, target_size)

    return img


print(check_model('20epochs.h5', '../denoising/andromeda.jpg' ))
