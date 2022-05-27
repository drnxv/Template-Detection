import numpy as np
import cv2 as cv

# paths of the video and template
video_path = '/Users/pranav/Documents/Programming/Projects/Template Detection/Video.mp4'
template_path = '/Users/pranav/Documents/Programming/Projects/Template Detection/cone template.png'

# constant methods provided by opencv in order to template match
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

# reading in all the necessary images 
template = cv.imread(template_path)
smaller_template = cv.resize(template, (80, 100))
smaller_gray = cv.cvtColor(smaller_template, cv.COLOR_BGR2GRAY)
# after resizing the template and converting to grayscale, we can easily
# obtain its dimensions 
width, height = smaller_gray.shape[::-1]

# this just shows the template
def show_template():
    tmp = cv.imread(template_path)
    cv.imshow('Template', tmp)
    cv.waitKey(0)
    cv.destroyAllWindows()

# this plays the original video
def show_video():
    video = cv.VideoCapture(video_path)
    while(video.isOpened()):
        ret, frame = video.read()
        if ret:
            cv.imshow('Output', frame)
            if cv.waitKey(25) & 0XFF == ord('q'):
                break
        else:
            break

    video.release()
    cv.waitKey(0)
    cv.destroyAllWindows()

# this performs the template matching in order to detect the traffic barrel
def template_matching():
    method = eval(methods[5])
    video = cv.VideoCapture(video_path)
    while(video.isOpened()):
        ret, frame = video.read()
        if ret:
            method = eval(methods[1])
            result = cv.matchTemplate(frame, smaller_template, method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            
            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc

            bottom_right = (top_left[0] + width, top_left[1] + height)
            cv.rectangle(frame, top_left, bottom_right, 255, 2)

            cv.imshow('Output', frame)
            if cv.waitKey(25) & 0XFF == ord('q'):
                break
        else:
            break

    video.release()
    cv.waitKey(0)
    cv.destroyAllWindows()

# runs the functions
if __name__ == '__main__':
    show_video()
    template_matching()
