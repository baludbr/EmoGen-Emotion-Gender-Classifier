import Preprocessing as pro
import DL_Models as my_models

X_train,y_train,X_test,y_test=pro.Data_Preprocessing()


# model = my_models.CNN_Model(X_train, y_train)
model=my_models.RNN_Model(X_train,y_train)

loss, acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Loss : {loss}")
print(f"Accuracy :{acc} ")
