import cv2
from PIL import Image
from numpy import savez_compressed
from numpy import asarray
import cv2
from numpy import expand_dims
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from utils import  utilities



def main():
    camera = cv2.VideoCapture(0)
    i = 0
    util_funcs = utilities()
    while True:
        return_value, image = camera.read()
        cv2.imwrite('temp/'+'opencv'+str(i)+'.png', image)
        test_images,x1s,y1s,widths,heights = util_funcs.extract_multiple_faces('temp/'+'opencv'+str(i)+'.png')    
        test_image = 'temp/'+'opencv'+str(i)+'.png'
        util_funcs.predict(test_image,test_images,x1s,y1s,widths,heights)    
        cv2.imwrite('temp/'+'opencv'+str(i)+'.png', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i+=1


    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
        