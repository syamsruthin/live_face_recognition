from os import listdir
from os.path import isdir
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import cv2
from numpy import load
from numpy import expand_dims
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle as pkl

class utilities:
    def __init__(self):                       
        with open('model_store/trained_models.pkl', 'rb') as input:            
            self.detector = MTCNN()
            self.in_encoder = pkl.load(input)
            self.out_encoder = pkl.load(input) 
            self.model = pkl.load(input)             
                
        self.facenet_model = load_model('keras-facenet/model/facenet_keras.h5') 

    def get_embedding(self,model, face_pixels):
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = model.predict(samples)
        return yhat[0]

    def predict(self,test_image_path,test_images,x1s,y1s,widths,heights):
        predict_names = []
        for test_image in test_images:            
            test_image_embedding = self.get_embedding(self.facenet_model,test_image)
            embeddings = list()
            embeddings.append(test_image_embedding)
            test_image_embedding = asarray(embeddings)
            in_encoder = Normalizer(norm='l2')
            test_image_embedding = in_encoder.transform(test_image_embedding)
            samples = expand_dims(test_image_embedding[0], axis=0)
            yhat_class = self.model.predict(samples)
            yhat_prob = self.model.predict_proba(samples)
            # get name
            class_index = yhat_class[0]
            class_probability = yhat_prob[0,class_index] * 100  
            image = cv2.imread(test_image_path)
            print(class_probability)
            predict_name = self.out_encoder.inverse_transform(yhat_class)     
            if class_probability < 60:
                predict_names.append('unknown')
            else:
                predict_names.append(predict_name[0])
        
        i = 0
        for x1,y1,width,height in zip(x1s,y1s,widths,heights):
            image = cv2.rectangle(image, (int(x1),int(y1)), (int(width),int(height)), (0, 255, 0), 2)            
            image = cv2.putText(image, predict_names[i], (x1,y1-3), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 1, cv2.LINE_AA)
            i+=1
            cv2.imshow('frame',image)


    def extract_multiple_faces(self,filename, required_size=(160, 160)):
        x1s,y1s,widths,heights = [],[],[],[]
        face_arrays = []
        image = Image.open(filename) #Load image
        image = image.convert("RGB") #Convert to Rgb
        pixels = asarray(image) #Convert to array
        results = self.detector.detect_faces(pixels) #Detect faces from given image
        # extract the bounding box from the first face
        #Extract coordinates for the detected faces (Will be list if there are more faces in given image)
        for result in results:
            x1, y1, width, height = result['box']        
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            x1s.append(x1),y1s.append(y1),widths.append(x2),heights.append(y2)
            # extract the face
            face = pixels[y1:y2, x1:x2]
            # resize pixels to the model size
            image = Image.fromarray(face)
            image = image.resize(required_size)
            face_array = asarray(image)
            face_arrays.append(face_array)
        return face_arrays,x1s,y1s,widths,heights

        