import os
import kfp
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics, ClassificationMetrics

@dsl.component(base_image="tensorflow/tensorflow")
def load_dataset(x_train_artifact: Output[Dataset], x_test_artifact: Output[Dataset],y_train_artifact: Output[Dataset],y_test_artifact: Output[Dataset]):
    '''
    get dataset from Keras and load it separating input from output and train from test
    '''
    import numpy as np
    from tensorflow import keras
    import os
    import shutil
   
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()    
    
    np.save("/tmp/x_train.npy",x_train)
    shutil.copy("/tmp/x_train.npy", x_train_artifact.path)
    
    np.save("/tmp/y_train.npy",y_train)
    shutil.copy("/tmp/y_train.npy", y_train_artifact.path)
    
    np.save("/tmp/x_test.npy",x_test)
    shutil.copy("/tmp/x_test.npy", x_test_artifact.path)
    
    np.save("/tmp/y_test.npy",y_test)
    shutil.copy("/tmp/y_test.npy", y_test_artifact.path)

@dsl.component(base_image="tensorflow/tensorflow")
def preprocessing(metrics : Output[Metrics], x_train_processed : Output[Dataset], x_test_processed: Output[Dataset],
                  x_train_artifact: Input[Dataset], x_test_artifact: Input[Dataset]):
    ''' 
    just reshape and normalize data
    '''
    import numpy as np
    import os
    import shutil
    
    # load data artifact store
    x_train = np.load(x_train_artifact.path) 
    x_test = np.load(x_test_artifact.path)
    
    # reshaping the data
    # reshaping pixels in a 28x28px image with greyscale, canal = 1. This is needed for the Keras API
    x_train = x_train.reshape(-1,28,28,1)
    x_test = x_test.reshape(-1,28,28,1)
    # normalizing the data
    # each pixel has a value between 0-255. Here we divide by 255, to get values from 0-1
    x_train = x_train / 255
    x_test = x_test / 255
    
    #logging metrics using Kubeflow Artifacts
    metrics.log_metric("Len x_train", x_train.shape[0])
    metrics.log_metric("Len y_train", x_test.shape[0])
   
    
    # save feuture in artifact store
    np.save("tmp/x_train.npy",x_train)
    shutil.copy("tmp/x_train.npy", x_train_processed.path)
    
    np.save("tmp/x_test.npy",x_test)
    shutil.copy("tmp/x_test.npy", x_test_processed.path)
    
@dsl.component(base_image="tensorflow/tensorflow")
def model_building(ml_model : Output[Model]):
    '''
    Define the model architecture
    This way it's more simple to change the model architecture and all the steps and indipendent
    '''
    from tensorflow import keras
    import tensorflow as tf
    import os
    import shutil
    
    #model definition
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28,28,1)))
    model.add(keras.layers.MaxPool2D(2, 2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))

    model.add(keras.layers.Dense(10, activation='softmax'))
    
    #saving model
    model.save("tmp/model.keras")
    shutil.copy("tmp/model.keras", ml_model.path)
    
@dsl.component(base_image="tensorflow/tensorflow", packages_to_install=['scikit-learn'])
def model_training(
    ml_model : Input[Model],
    x_train_processed : Input[Dataset], x_test_processed: Input[Dataset],
    y_train_artifact : Input[Dataset], y_test_artifact :Input[Dataset],
    hyperparameters : dict, 
    metrics: Output[Metrics], classification_metrics: Output[ClassificationMetrics], model_trained: Output[Model]
    ):
    """
    Build the model with Keras API
    Export model metrics
    """
    from tensorflow import keras
    import tensorflow as tf
    import numpy as np
    import os
    import glob
    from sklearn.metrics import confusion_matrix
    import shutil
    
    #load dataset
    x_train = np.load(x_train_processed.path)
    x_test = np.load(x_test_processed.path)
    y_train = np.load(y_train_artifact.path)
    y_test = np.load(y_test_artifact.path)
    
    #load model structure
    shutil.copy(ml_model.path, "tmp/ml_model.keras")
    model = keras.models.load_model("tmp/ml_model.keras")
    
    #reading best hyperparameters from katib
    lr=float(hyperparameters["lr"])
    no_epochs = int(hyperparameters["num_epochs"])
    
    #compile the model - we want to have a binary outcome
    model.compile(tf.keras.optimizers.SGD(learning_rate=lr),
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

    
    #fit the model and return the history while training
    history = model.fit(
      x=x_train,
      y=y_train,
      epochs=no_epochs,
      batch_size=20,
    )

     
    # Test the model against the test dataset
    # Returns the loss value & metrics values for the model in test mode.
    model_loss, model_accuracy = model.evaluate(x=x_test,y=y_test)
    
    #build a confusione matrix
    y_predict = model.predict(x=x_test)
    y_predict = np.argmax(y_predict, axis=1)
    cmatrix = confusion_matrix(y_test, y_predict)
    cmatrix = cmatrix.tolist()
    numbers_list = ['0','1','2','3','4','5','6','7','8','9']
    #log confusione matrix
    classification_metrics.log_confusion_matrix(numbers_list,cmatrix)
  
    #Kubeflox metrics export
    metrics.log_metric("Test loss", model_loss)
    metrics.log_metric("Test accuracy", model_accuracy)
    
    #adding /1/ subfolder for TFServing and saving model to artifact store
    # model_trained.uri = model_trained.uri + '/1/'
    keras.models.save_model(model, "tmp/model_trained.keras")
    shutil.copy("tmp/model_trained.keras", model_trained.path)
    
@dsl.pipeline(
    name='mnist-classifier-dev',
    description='Detect digits')
def mnist_pipeline(hyperparameters: dict):
    load_task = load_dataset()
    preprocess_task = preprocessing(
        x_train_artifact = load_task.outputs["x_train_artifact"],
        x_test_artifact = load_task.outputs["x_test_artifact"]
    )
    model_building_task = model_building()
    training_task = model_training(
        ml_model = model_building_task.outputs["ml_model"],
        x_train_processed = preprocess_task.outputs["x_train_processed"],
        x_test_processed = preprocess_task.outputs["x_test_processed"],
        y_train_artifact = load_task.outputs["y_train_artifact"],
        y_test_artifact = load_task.outputs["y_test_artifact"],
        hyperparameters = hyperparameters
    )
    
    
#setting up the client
with open(os.environ['KF_PIPELINES_SA_TOKEN_PATH'], "r") as f:
    TOKEN = f.read()
client = kfp.Client(
    existing_token=TOKEN,
    host='https://ds-pipeline-dspa-test-pipelines.apps.cluster-wfsj7.sandbox1839.opentlc.com',
)

#running parameters
hyperparameters ={"hyperparameters" :  {"lr":0.1, "num_epochs":1 } }
namespace="test-pipelines"

client.create_run_from_pipeline_func(mnist_pipeline, arguments=hyperparameters,experiment_name="test",namespace="test-pipelines",enable_caching=False)