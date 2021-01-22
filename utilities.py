
import pandas as pd
import numpy as np
import seaborn as sns
import cv2
import tensorflow as tf
import os 
from skimage import io
from PIL import Image
from tensorflow.keras import backend as K
  
#creating a custom datagenerator:

class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, ids , list_class, list_rle, image_dir, batch_size = 16, img_h = 256, img_w = 256, shuffle = True):

    self.ids = ids
    self.class_type = list_class
    self.rle = list_rle
    self.image_dir = image_dir
    self.batch_size = batch_size
    self.img_h = img_h
    self.img_w = img_w
    self.shuffle = shuffle
    self.on_epoch_end()

  def __len__(self):
    'Get the number of batches per epoch'

    return int(np.floor(len(self.ids)) / self.batch_size)

  def __getitem__(self, index):
    'Generate a batch of data'

    #generate index of batch_size length
    indexes = self.indexes[index* self.batch_size : (index+1) * self.batch_size]

    #get the ImageId corresponding to the indexes created above based on batch size
    list_ids = [self.ids[i] for i in indexes]

    #get the ClassId corresponding to the indexes created above based on batch size
    list_class = [self.class_type[i] for i in indexes]

    #get the rle corresponding to the indexes created above based on batch size
    list_rle = [self.rle[i] for i in indexes]

    #generate data for the X(features) and y(label)
    X, y = self.__data_generation(list_ids, list_class, list_rle)

    #returning the data
    return X, y

  def on_epoch_end(self):
    'Used for updating the indices after each epoch, once at the beginning as well as at the end of each epoch'
    
    #getting the array of indices based on the input dataframe
    self.indexes = np.arange(len(self.ids))

    #if shuffle is true, shuffle the indices
    if self.shuffle:
      np.random.shuffle(self.indexes)

  def __data_generation(self, list_ids, list_class, list_rle):
    'generate the data corresponding the indexes in a given batch of images'

    # create empty arrays of shape (batch_size,height,width,depth) 
    #Depth is 1 for input and depth is taken as 4 for output becasue of 4 types of defect
    X = np.empty((self.batch_size, self.img_h, self.img_w, 1))
    y = np.empty((self.batch_size, self.img_h, self.img_w, 4))

    #iterate through the dataframe rows, whose size is equal to the batch_size
    for index, id in enumerate(list_ids):
      #path of the image
      path = 'train_images/' + str(id)

      #reading the image along blue channel(0)
      img = io.imread(path)
      img =img[:,:,0]

      #resizing the image and coverting them to array of type float64
      img = cv2.resize(img,(self.img_h,self.img_w))
      img = np.array(img, dtype = np.float64)

      #standardising the image
      img -= img.mean()
      img /= img.std()

      #creating a empty mask for label
      mask = np.empty((self.img_h,self.img_w,4))

      #iterating through the 4 class id to creat mask for each defect of the same image
      for j, class_id in enumerate([1,2,3,4]):

        #getting rle from list, using index used to get the image id
        rle = list_rle[index]

        #create a mask using rle if it belongs to a class_id else create mask with zeros if there no rle belonging to a class_id
        if list_class[index] == class_id:
          class_mask = rle2mask(rle,256,1600)
        else:
          class_mask = np.zeros((256,1600))
        
        #resizing the mask to shape (256,256)
        resized_mask = cv2.resize(class_mask,(self.img_h,self.img_w))

        #adding mask corresponding to each class_id
        mask[...,j] = resized_mask 
      
      #expanding the dimnesion of the image from (256,256) to (256,256,1)
      X[index,] = np.expand_dims(img, axis = 2)
      y[index,] = mask
    
    #normalizing y
    y = (y > 0).astype(int)

    return X, y



#function to convert rle to mask

def rle2mask(rle, height, width):
  

  #creating a one dimentional array containing 0's of length obtained by multiplying height and width of the image
  mask = np.zeros(height*width).astype(np.uint8)

  #spliting the rle based on space , try running the rle.split() on separate cell to see how the values are separated based on space
  rle = rle.split()

  #selecting every second value in the list like obtaining values corresponding to indexes 0,2,4,....
  start = rle[0::2]

  #selecting every second value in the list like obtaining values corresponding to indexes 1,3,5,....
  length = rle[1::2]

  '''
  For example if rle value looks like this '4954 7 5800 20', in such a case elements belonging to even index like 4954, 5800 
  are taken as start point and the values belong to odd index likr 7,20 are the length. So we need to add length to the respective
  start pixels like 4954+7 and 5800+20, to the ending point. Now, we need to  apply mask '1' in pixles from 4954 to 4961 and 5800 to 5820.
  '''
  
  for i in range(len(start)):
    mask[int(start[i]):(int(start[i])+int(length[i]))] = 1
  
  #Now the shape of the mask is in one dimension, we need to convert the mask to the same dimension as the image, initally using reshape and followed by Transpose
  img = mask.reshape(width, height)
  img = img.T
  return img
  
  


#Function to convert mask to rle

def mask2rle(mask):

  #We do the reverse of what we did in the above function, initially apply Transpose to the mask image and then flatten to one dimension
  pixels = mask.T.flatten()

  #Here, we add extra values at front and end , this would help in finding the correct length of pixels that have been masked
  pixels = np.concatenate([[0], pixels, [0]])

  '''
  Here, consider a array which is like [0,0,1,1,1,0], we added two values at the first and last of the array, now the array looks like
  [0,0,0,1,1,1,0,0]. After that, we check whether pixels[1:] != pixels[:-1], (i.e.)[0,0,1,1,1,0,0] != [0,0,0,1,1,1,0]. As you can see, if we
  compared these two we would get something like[False,False,True,False,False,True,False]. Then,using np.where, we will get indexes
  corresponding to True, in this case, 2 and 5
  '''
  rle = np.where(pixels[1:] != pixels[:-1])[0]

  #here we subtract values in even index in the obtained list(i.e) [2,5], from the odd index, (i.e.)5-2 = 3, Now the list would look like [2, 3]
  rle[1::2] -= rle[0::2]
  
  #finally join to rle format, in this case it would look like ('2 3')
  return ' '.join(str(x) for x in rle)



def prediction(test, model, model_seg):
  '''
  Predcition function which takes dataframe containing ImageID as Input and perform 2 type of prediction on the image
  Initially, image is passed through the classification network which predicts whether the image has defect or not, if the model
  is 99% sure that the image has no defect, then the image is labeled as no-defect, if the model is not sure, it passes the image to the
  segmentation network, it again checks if the image has defect or not, if it has defect, then the type and location of defect is found
  '''

  #directory
  directory = "train_images"

  #Creating empty list to store the results
  mask = []
  defect_type = []
  image_id = []

  #iterating through each image in the test data
  for i in test.ImageId:

    path = os.path.join(directory, i)

    #reading the image
    img = io.imread(path)

    #Normalizing the image
    img = img * 1./255.

    #Reshaping the image
    img = cv2.resize(img,(256,256))

    #Converting the image into array
    img = np.array(img, dtype = np.float64)
    
    #reshaping the image from 256,256,3 to 1,256,256,3
    img = np.reshape(img, (1,256,256,3))

    #making prediction on the image
    defect_or_no_defect = model.predict(img)

    #if the image has defect we append the details of the image to the list
    if defect_or_no_defect < 0.01:
      image_id.append(i)
      defect_type.append(0)
      mask.append('0 0')
      continue

    #reading the image along blue channel (0)( we can take any channel or 3 channel as itself, if we are taking 3 channels we need to change X depth to 3)
    img = io.imread(path)
    img =img[:,:,0]

    #Creating a empty array of shape 1,256,256,1
    X = np.empty((1, 256, 256, 1))

    #resizing the image and coverting them to array of type float64
    img = cv2.resize(img,(256,256))
    img = np.array(img, dtype = np.float64)

    #standardising the image
    img -= img.mean()
    img /= img.std()

    #converting the shape of image from 256,256 to 1,256,256,1
    X[0,] = np.expand_dims(img, axis = 2)

    #make prediction
    predict = model_seg.predict(X)

    #if the sum of predicted values is equal to 0 then there is no defect
    if predict.round().astype(int).sum() == 0:
      image_id.append(i)
      defect_type.append(0)
      mask.append('0 0')
      continue

    #iterating 4 times to get the prediction of 4 different classes
    for j in range(4):
      #since j values through iteration are 0,1,2,3 , we add 1 to j to make it as classIDs corresponding to 1,2,3,4
      class_id = j + 1

      #get the mask values of each class
      mask_value = predict[0,:,:,j].round().astype(int)

      #if the sum of mask values is greater than 0.5(anything greater than 0 ), that class has defect
      if mask_value.sum() > 0.5:
        try:
            #applying mask to image, area with defect will be highlighted in white(255)
            img[mask_value == 1] = 255
            #since our original shape is 256,1600, we reshape to that size
            img = cv2.resize(img,(1600,256))
            #Now, we mask the image such that, areas which are not white(defected areas) to be black(0)
            img[img < 255] = 0
            #we again normalize the values
            img = img * 1./255.
            #get the rle for the create masked image
            rle = mask2rle(img)
        except:
            continue

        #append the valeues to the respective listes
        image_id.append(i)
        defect_type.append(class_id)
        mask.append(rle)

  return image_id, defect_type, mask
        




'''
We need a custom loss function to train this ResUNet.So,  we have used the loss function as it is from https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py


@article{focal-unet,
  title={A novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation},
  author={Abraham, Nabila and Khan, Naimul Mefraz},
  journal={arXiv preprint arXiv:1810.07842},
  year={2018}
}
'''
def tversky(y_true, y_pred, smooth = 1e-6):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)