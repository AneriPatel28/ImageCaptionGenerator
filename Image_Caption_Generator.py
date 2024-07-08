# # Import Libraries 

# In[1]:


import numpy as np
from PIL import Image
import os
import string
from pickle import dump
from pickle import load
from keras.applications.xception import Xception #to get pre-trained model Xception
from keras.applications.xception import preprocess_input
import keras
from keras.preprocessing import image 
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer #for text tokenization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tensorflow.keras.layers import add
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.layers import LSTM, Embedding, Dropout
from tqdm import tqdm_notebook as tqdm #to check loop progress


# In[2]:


tqdm().pandas()


# # Data Cleaning

# File reading

# In[4]:


#First we will load the file in the memory
def loadFile(filename):
    file=open(filename,'r') #open file
    context= file.read() #read file and store it in context
    file.close()
    return context


# Getting Image name and descriptions in form of dictionary

# In[5]:


# Get all images with their captions
def imageCaptions(filename):
    file = loadFile(filename)
    captions = file.split('\n')# We are splitting with a new line and storing it in list of captions
    descriptions ={}
    for caption in captions[:-1]:
        image_name, caption = caption.split('\t')#we are splitting each caption with image no. and their respective caption
        #print("IMAGE ",img)
        #print('CAPTIONS ', caption)
        if image_name[:-2] not in descriptions: # we are checking that is the name of image except its caption, no. is present in description or not.
            descriptions[image_name[:-2]] = [ caption ] #if no then we will make that image name key and assign the caption to its key value in list form.
        else:
            descriptions[image_name[:-2]].append(caption) # if name is already present then we will just append the caption in a list.

    return descriptions


# Cleaning the Image descriptions

# In[6]:


def dataCleaning(captions):
    table = str.maketrans('','',string.punctuation)
    for image_name,caps in captions.items():
    #print(img)#name of image
    #print(caps) # list of captions
          for i,image_caption in enumerate(caps):
            #print(i) #number
              #print(image_caption) #caption
            image_caption.replace("-"," ")
              #print("After replacing : ###### ", image_caption)
            list_of_word = image_caption.split() # we are making list of words in each caption
              
              #converts to lowercase
            list_of_word = [word.lower() for word in list_of_word]

              #print(list_of_word)
                
              #remove punctuation from each token
            list_of_word = [word.translate(table) for word in list_of_word]
              #print(list_of_word)

              #remove hanging 's and a 
            list_of_word = [word for word in list_of_word if(len(word)>1)]
              #print(list_of_word)


              #remove tokens with numbers in them
            list_of_word = [word for word in list_of_word if(word.isalpha())] #check that if each character in a word is b/w a-z
              #print(list_of_word)

            image_caption = ' '.join(list_of_word) #joining the list of words
              #print(">>>>>>",image_caption)

            descriptions[image_name][i]= image_caption #now assigning each caption after cleaning with its image name
              #print("@@@@@",descriptions)
    return descriptions



# Building vocabulary

# In[7]:


def vocabulary(descriptions):
    # build vocabulary of all unique words
    vocabul = set()
    for key in descriptions.keys():
        [vocabul.update(d.split()) for d in descriptions[key]] #adds elements from a set (passed as an argument) to the set. 
    return vocabul


# Storing captions in one file

# In[8]:


def saveImageDescriptions(descriptions, filename):
    lines = list()
    for image_name, caption_list in descriptions.items():
        for caption in caption_list:
            lines.append(image_name + '\t' + caption ) #concenating image name and caption
    data = "\n".join(lines)
    file = open(filename,"w")
    file.write(data)
    file.close()


# Apply all functions that  we created

# In[9]:


descriptions = imageCaptions('Flickr8k_text/Flickr8k.token.txt')
print("Length of descriptions =" ,len(descriptions))


# In[10]:


data_clean=dataCleaning(descriptions)


# In[11]:


data_clean


# In[12]:


#building vocabulary 
vocabulary = vocabulary(data_clean)
print("Length of vocabulary = ", len(vocabulary))


# In[13]:


#saving each description to file 
saveImageDescriptions(data_clean, " descriptions.txt")


# # Feature Extraction

# In[15]:


print(len(os.listdir('Flicker8k_Dataset')))


# In[16]:


def featureExtraction(path):
    model = Xception( include_top=False, pooling='avg' )
    features = {}
    for image in tqdm(os.listdir(path)):
            filename = path+ "/" + image
            images = Image.open(filename)
            #print(images)
            images = images.resize((299,299))
            #print(images)
            images = np.expand_dims(images, axis=0)
            #print(images)
            
        
            images = images/127.5
            images = images - 1.0

            feature = model.predict(images)
            features[image] = feature
    return features


# In[17]:


print(len(os.listdir('Flicker8k_Dataset')))


# In[23]:


images_dataset='Flicker8k_Dataset'
features = featureExtraction(images_dataset)
dump(features, open("Features.p","wb"))


# In[24]:


features=load(open("features.p","rb"))


# In[25]:


len(features)


# In[ ]:





# # Loading dataset for training model

# Load photos

# In[31]:


def photoLoading(filename):
    file = loadFile(filename)
    photos_name = file.split("\n")[:-1]
    return photos_name


# Loading clean description

# In[32]:


def cleanDataLoading(filename, photos_name): 
    #loading clean_descriptions
    file = loadFile(filename)
    descriptions = {}
    for line in file.split("\n"):
        words = line.split()
        if len(words)<1 :
            continue
        image, image_caption = words[0], words[1:]
        if image in photos_name:
            if image not in descriptions:
                descriptions[image] = []
            caption = '<start> ' + " ".join(image_caption) + ' <end>'
            descriptions[image].append(caption)
    return descriptions


# Loading Features

# In[33]:


def featureLoading(photos_name):
    #loading all features
    allFeatures = load(open("features.p","rb"))
    #selecting only needed features
    features = {k:allFeatures[k] for k in photos_name}
    return features


# Applying all the functions

# In[36]:


fileName = 'Flickr8K_text/Flickr_8k.trainImages.txt'


# In[37]:


#train = loading_data(filename)
training_data_images = photoLoading(fileName)


# In[38]:


len(training_data_images)


# In[42]:


training_data_descriptions = cleanDataLoading(" Descriptions.txt", training_data_images)


# In[43]:


training_data_features = featureLoading(training_data_images)


# #Tokenizing the vocabulary 

# Converting the dictionary to list

# In[44]:


def dict2list(descriptions):
    all_captions = []
    for key in descriptions.keys():
        [all_captions.append(d) for d in descriptions[key]]
    return all_captions


# Creating tokenizer class

# In[45]:


def creatingTokenizer(descriptions):
    captions_list = dict2list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions_list)
    return tokenizer


# Applying all functions

# In[46]:


tokenizer = creatingTokenizer(training_data_descriptions)
dump(tokenizer, open('Tokens.p', 'wb'))
size_vocabulary = len(tokenizer.word_index) + 1
size_vocabulary


# In[47]:


tokenizer


# #Finding the maximum length of descriptions
# 

# In[48]:


def maximumLength(descriptions):
    caption_list = dict2list(descriptions)
    return max(len(d.split()) for d in caption_list)
    
maximumLength = maximumLength(descriptions)
maximumLength


# In[ ]:





# # Create Data Generator
# 

# Creating Sequence

# In[49]:


def create_sequences(creatingTokenizer, maximumLength, caption_list, features):
    feature_vector, Text_seq, word_predict = list(), list(), list()
    # walk through each description for the image
    for caption in caption_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([caption])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            input_seq, output_seq = seq[:i], seq[i]
            # pad input sequence
            
            #Pads sequences to the same length. This function transforms a list (of length num_samples ) of 
            #sequences (lists of integers) into a 2D Numpy array of shape (num_samples, num_timesteps) 
            
            input_seq = pad_sequences([input_seq], maxlen=maximumLength)[0]
            
            # encode output sequence
            output_seq = to_categorical([output_seq], num_classes=size_vocabulary)[0]
            # store
            feature_vector.append(features)
            Text_seq.append(input_seq)
            word_predict.append(output_seq)
    return np.array(feature_vector), np.array(Text_seq), np.array(word_predict)


# In[50]:


create_sequences


# In[51]:


#create input-output sequence pairs from the image description.
#data generator, used by model.fit_generator()
def dataGenerator(descriptions, features, tokenizer, maximumLength):
    while 1:
        for key, description_list in descriptions.items():
            #retrieve photo features
            feature = features[key][0]
            input_image, input_sequence, output_word = create_sequences(tokenizer, maximumLength, description_list, feature)
            yield [[input_image, input_sequence], output_word]


# In[ ]:





# In[ ]:





# # Defining Model

# In[52]:


from keras.utils import plot_model


# In[53]:


def modelDefining(size_vocabulary, maximumLength):
    # features from the CNN model squeezed from 2048 to 256 nodes
    inputs1 = Input(shape=(2048,))
    feature_ex_1 = Dropout(0.5)(inputs1)#The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting.
    feature_ex_2 = Dense(256, activation='relu')(feature_ex_1)

    # LSTM sequence model
    inputs2 = Input(shape=(maximumLength,))
    sequence_1 = Embedding(size_vocabulary, 256, mask_zero=True)(inputs2)
    sequence_2 = Dropout(0.5)(sequence_1)
    sequence_3 = LSTM(256)(sequence_2)

        # Merging both models
    decoder1 = add([feature_ex_2, sequence_3])
        #Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the 
        #element-wise activation function passed as the activation argument, kernel is a weights matrix created 
        #by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True).
        # These are all attributes of Dense.

    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(size_vocabulary, activation='softmax')(decoder2)

        # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')


    # summarize model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True) #Converts a Keras model to dot format and save to a file.


    return model


# In[58]:


model = modelDefining(size_vocabulary, maximumLength)
epochs = 12
steps = len(training_data_descriptions)
# making a directory models to save our models
os.mkdir("MODELS")
for i in range(epochs):
    generator = dataGenerator(training_data_descriptions, training_data_features, tokenizer, maximumLength)
    model.fit_generator(generator, epochs=1, steps_per_epoch= steps, verbose=1)
    model.save("MODELS/model_" + str(i) + ".h5")


# Training Data

# In[59]:


print('Dataset: ', len(training_data_images))
print('Descriptions: train=', len(training_data_descriptions))
print('Photos: train=', len(training_data_features))
print('Vocabulary Size:', size_vocabulary)
print('Description Length: ', maximumLength)


# # Testing Model

# In[3]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
#argparse — parse the arguments. Using argparse is how we let the user of our program provide values for 
#variables at runtime. It's a means of communication between the writer of a program and the user


# In[4]:


def extract_features(filename, model):
        
        image = Image.open(filename)
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature


# In[5]:


def index2word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# In[6]:


def captionGenerator(model, tokenizer, photo, maximumLength):
    text = 'start'
    for i in range(maximumLength):
        pattern = tokenizer.texts_to_sequences([text])[0]
        pattern = pad_sequences([pattern], maxlen=maximumLength)
        predicted_seq = model.predict([photo,pattern], verbose=0)
        predicted_seq = np.argmax(predicted_seq)
        word = index2word(predicted_seq, tokenizer)
        if word is None:
            break
        text += ' ' + word
        if word == 'end':
            break
    return text


# In[7]:


def load_clean_descriptions(filename, dataset):
    # load document
    doc = loadFile(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if image_id in dataset:
            # create list
            if image_id not in descriptions:
                descriptions[image_id] = list()
            # wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            # store
            descriptions[image_id].append(desc)
    return descriptions
 


# In[8]:


def load_photo_features(filename, dataset):
    # load all features
    all_features = load(open(filename, 'rb'))
    # filter features
    features = {k: all_features[k] for k in dataset}
    return features


# In[9]:


maximumLength = 32
tokenizer = load(open("Tokens.p","rb"))
model = load_model('MODELS/model_11.h5')



def display_output(IMAGE):
        image_path=IMAGE
        xception_model = Xception(include_top=False, pooling="avg")
        photo = extract_features(image_path,xception_model)
        description = captionGenerator(model, tokenizer, photo, maximumLength)
        #print("\n\n")
        #print(description)
        img = Image.open(image_path)
        #plt.imshow(img)
        return description


# # GUI

# In[10]:



import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog


# In[11]:


def upload_img():
    global img, image_data
    for img_display in frame.winfo_children():
        img_display.destroy()

    image_data = filedialog.askopenfilename(initialdir="/", title="Choose an image",
                                       filetypes=(("all files", "*.*"), ("jpg files", "*.jpg")))
    basewidth = 300
    img = Image.open(image_data)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    file_name = image_data.split('/')
    panel = tk.Label(frame, text= str(file_name[len(file_name)-1]).upper()).pack()
    panel_image = tk.Label(frame, image=img).pack()
    
def caption():
    
    caption=display_output(image_data)

    table = tk.Label(frame, text="Caption: " + caption[6:-4], font=("Helvetica", 12)).pack()


# In[ ]:


root = tk.Tk()
root.title('IMAGE CAPTION GENERATOR')
root.iconbitmap('class.ico')
root.resizable(False, False)
tit = tk.Label(root, text="IMAGE CAPTION GENERATOR", padx=25, pady=6, font=("", 12)).pack()
canvas = tk.Canvas(root, height=550, width=600, bg='#355c7d')
canvas.pack()
frame = tk.Frame(root, bg='white')
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
chose_image = tk.Button(root, text='Choose Image',
                        padx=35, pady=10,
                        fg="black", bg="pink", command=upload_img, activebackground="#add8e6")
chose_image.pack(side=tk.LEFT)

caption_image = tk.Button(root, text='Caption Image',
                        padx=35, pady=10,
                        fg="black", bg="pink", command=caption, activebackground="#add8e6")
caption_image.pack(side=tk.RIGHT)
root.mainloop()

