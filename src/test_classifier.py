from Tools import *

cornerClassifier = Classifier()
cornerClassifier.train_classifier('corner')
observation = cornerClassifier.data[5]
proba = cornerClassifier.classifier.predict_proba([observation.data_filled[:,0]])
label = (cornerClassifier.classifier.classes_[np.argmax(proba)])
print(proba)
print(label)