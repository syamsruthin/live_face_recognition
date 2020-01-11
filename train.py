from os import listdir
from os.path import isdir
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import cv2
from numpy import load
from numpy import expand_dims
from numpy import savez_compressed
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle as pkl



# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename) #Load image
    image = image.convert("RGB") #Convert to Rgb
    pixels = asarray(image) #Convert to array
    results = detector.detect_faces(pixels) #Detect faces from given image
    # extract the bounding box from the first face
    #Extract coordinates for the detected faces (Will be list if there are more faces in given image)
    x1, y1, width, height = results[0]['box'] 
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array,x1,y1,x2,y2

def load_faces(directory):
    faces = list()
    for filename in listdir(directory):
        path = directory+filename
        face,_,_,_,_ = extract_face(path)
        faces.append(face)
    return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
    # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

def get_embedding(model, face_pixels):
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

detector = MTCNN()

if __name__ == "__main__":
    # load train dataset
    trainX, trainy = load_dataset('data/train/')
    print(trainX.shape, trainy.shape)
    # load test dataset
    testX, testy = load_dataset('data/val/')
    # save arrays to one file in compressed format
    savez_compressed('model_store/to-train-faces-dataset.npz', trainX, trainy, testX, testy)
    # load the face dataset
    #data = load('model_store/5-celebrity-faces-dataset.npz')
    #trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    #print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
    # load the facenet model
    model = load_model('keras-facenet/model/facenet_keras.h5')
    print('Loaded Model')
    # convert each face in the train set to an embedding
    newTrainX = list()
    for face_pixels in trainX:
        embedding = get_embedding(model, face_pixels)
        newTrainX.append(embedding)
    newTrainX = asarray(newTrainX)
    print(newTrainX.shape)
    # convert each face in the test set to an embedding
    newTestX = list()
    for face_pixels in testX:
        embedding = get_embedding(model, face_pixels)
        newTestX.append(embedding)
    newTestX = asarray(newTestX)
    print(newTestX.shape)
    # save arrays to one file in compressed format
    savez_compressed('model_store/trained-faces-embeddings.npz', newTrainX, trainy, newTestX, testy)
    data = load('model_store/trained-faces-embeddings.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)
    # fit model
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)
    # predict
    yhat_train = model.predict(trainX)
    yhat_test = model.predict(testX)
    # score
    score_train = accuracy_score(trainy, yhat_train)
    score_test = accuracy_score(testy, yhat_test)
    # summarize
    print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

    with open('model_store/trained_models.pkl','wb') as output:        
        pkl.dump(in_encoder,output)
        pkl.dump(out_encoder,output)
        pkl.dump(model,output)