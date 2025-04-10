import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob

from keras.utils import image_dataset_from_directory
from keras import layers
from keras import Sequential
from keras.optimizers.legacy import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.regularizers import L2

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight
from sklearn.model_selection import KFold

path_train = './chest_xray/train'
path_test = './chest_xray/test'

img_dim = 64

img_size = (img_dim, img_dim)
batch_size = 64

Xtrain, Xval = image_dataset_from_directory(path_train, image_size=img_size, batch_size=batch_size,
                                            validation_split=0.2, subset="both", seed=123)
Xtest = image_dataset_from_directory(path_test, image_size=img_size, batch_size=batch_size, seed=123)

_, _, files_normal = next(os.walk("./chest_xray/train/NORMAL"))
normal_num = len(files_normal)

_, _, files_pneumonia = next(os.walk("./chest_xray/train/PNEUMONIA"))
pneumonia_num = len(files_pneumonia)

classes = Xtrain.class_names
print(classes)

out_num = np.append(np.zeros((normal_num, 1)), np.ones((pneumonia_num, 1)))
out_class = []

plt.figure()
plt.hist(out_num)
plt.grid()
plt.title("K0 - NORMAL, K1 - PNEUMONIA")
plt.show()

for i in range(out_num.size):
    if out_num[i] == 0:
        out_class.append("NORMAL")
    else:
        out_class.append("PNEUMONIA")

for filename in glob.glob('./chest_xray/train/NORMAL/*.jpeg'):
    img = Image.open(filename)
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title('Primer zdravog pacijenta')
    plt.axis('off')
    plt.show()
    break

for filename in glob.glob('./chest_xray/train/PNEUMONIA/*.jpeg'):
    img = Image.open(filename)
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title('Primer pacijenta sa pneumonijom')
    plt.axis('off')
    plt.show()
    break

data_augmentation = Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(img_size[0], img_size[1], 3)),
        layers.RandomRotation(0.25),
        layers.RandomZoom(0.1),
    ]
)

num_classes = len(classes)

model = Sequential([
    data_augmentation,
    layers.Rescaling(1. / 255, input_shape=(img_dim, img_dim, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu', kernel_regularizer=L2(0.001)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=L2(0.001)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=L2(0.001)),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=L2(0.001)),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(Adam(learning_rate=0.001), loss=SparseCategoricalCrossentropy(), metrics='accuracy')

weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(classes), y=out_class)
model.summary()

lookup_images = {}
lookup_labels = {}
X = Xtrain.concatenate(Xval)
for i, (x, y) in enumerate(X):
    lookup_images[i] = x
    lookup_labels[i] = y

kf = KFold(n_splits=5, shuffle=True, random_state=20)

fold_no = 1
max_acc = 0
acc_all = []

for trening, val in kf.split(np.arange(len(X))):
    images_train = np.concatenate(list(map(lookup_images.get, trening)))
    labels_train = np.concatenate(list(map(lookup_labels.get, trening)))

    images_val = np.concatenate(list(map(lookup_images.get, val)))
    labels_val = np.concatenate(list(map(lookup_labels.get, val)))

    history = model.fit(images_train, labels_train, epochs=50, validation_data=[images_val, labels_val], verbose=0,
                        class_weight={0: weights[0], 1: weights[1]})

    labels = np.array([])
    pred = np.array([])
    for img, lab in Xval:
        labels = np.append(labels, lab)
        pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))

    acc = accuracy_score(labels, pred)
    acc_all.append(acc)

    print(f'Tacnost u {fold_no}. fold-u je: {100 * acc}%')

    if acc > max_acc:
        max_acc = acc
        best_model = model
        best_model_history = history

    fold_no += 1

print(f'Prosecna tacnost modela postupkom krosvalidacije je: {100 * np.mean(acc_all)}%')

acc = best_model_history.history['accuracy']
val_acc = best_model_history.history['val_accuracy']

loss = best_model_history.history['loss']
val_loss = best_model_history.history['val_loss']

plt.figure()
plt.subplot(121)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.subplot(122)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
plt.show()

labels = np.array([])
pred = np.array([])
case_TP = 0
case_TN = 0
case_FP = 0
case_FN = 0
cnt = 0

for img, lab in Xtest:
    labels = np.append(labels, lab)
    pred = np.append(pred, np.argmax(best_model.predict(img, verbose=0), axis=1))
    if (labels[cnt] == 0 and pred[cnt] == 0 and case_TN == 0):
        plt.figure()
        plt.imshow(img[cnt].numpy().astype('uint8'))
        plt.title('Primer dobro klasifikovanog zdravog pacijenta')
        plt.axis('off')
        plt.show()
        case_TP = 1
    if (labels[cnt] == 0 and pred[cnt] == 1 and case_FP == 0):
        plt.figure()
        plt.imshow(img[cnt].numpy().astype('uint8'))
        plt.title('Primer lose klasifikovanog zdravog pacijenta')
        plt.axis('off')
        plt.show()
        case_FP = 1
    if (labels[cnt] == 1 and pred[cnt] == 0 and case_FN == 0):
        plt.figure()
        plt.imshow(img[cnt].numpy().astype('uint8'))
        plt.title('Primer lose klasifikovanog bolesnog pacijenta')
        plt.axis('off')
        plt.show()
        case_FN = 1
    if (labels[cnt] == 1 and pred[cnt] == 1 and case_TN == 0):
        plt.figure()
        plt.imshow(img[cnt].numpy().astype('uint8'))
        plt.title('Primer dobro klasifikovanog bolesnog pacijenta')
        plt.axis('off')
        plt.show()
        case_TN = 1
    cnt = cnt + 1

print("Tacnost modela na test skupu je: ", str(100 * accuracy_score(labels, pred)) + '%')

cm = confusion_matrix(labels, pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot()
plt.title('Matrica konfuzije na test skupu')
plt.show()


for img, lab in Xtrain:
    labels = np.append(labels, lab)
    pred = np.append(pred, np.argmax(best_model.predict(img, verbose=0), axis=1))

cm = confusion_matrix(labels, pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot()
plt.title('Matrica konfuzije na trening skupu')
plt.show()







