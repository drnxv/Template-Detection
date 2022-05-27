import numpy as np
import cv2 as cv

template_path = '/Users/pranav/Documents/Programming/Projects/Template Detection/tiger_face.png'
image_path = '/Users/pranav/Documents/Programming/Projects/Template Detection/tiger_image.jpeg'
# methods found in the opencv module which let us detect objects
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

# read in the image
image = cv.imread(image_path)
# read in the template
template = cv.imread(template_path)
# resize the template in order to template matching to be more effective
smaller_template = cv.resize(template, (200, 200))
# convert the smaller template into grayscale in order to get dimensions easier
smaller_gray = cv.cvtColor(smaller_template, cv.COLOR_BGR2GRAY)
# getting the dimensions of the smaller template
smaller_width, smaller_height = smaller_gray.shape[::-1]

# function which shows the entire image
def show_image():
    cv.imshow("Tiger", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# function which shows the template
def show_template():
    cv.imshow("Tiger Face", template)
    cv.waitKey(0)
    cv.destroyAllWindows()

# function which is going to detect the face
def face_detection():
    # choosing the first method
    method = eval(methods[0])
    result = cv.matchTemplate(image, smaller_template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + smaller_width, top_left[1] + smaller_height)
    cv.rectangle(image, top_left, bottom_right, 255, 2)

    cv.imshow("Facial Detection", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    show_image()
    face_detection()
