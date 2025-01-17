# Handwritten Digit Classifier

# Warm Cool Prediction
Application description
## Description
This application is a simple image classifier that predicts the digit on the image trained using MNIST dataset. 
The application uses opencv to predict the image. 
The application is built using OpenCV and Flask framework. 
The flask application runs on http://127.0.0.1:8080
The user can upload an image to the application, and the application will predict the digit on the image.
The files will be uploaded to the images folder in the application directory.

## Installation
Run the app from the terminal using the following commands:
```bash
# on terminal
# navigate to the project path
cd /path/to/your/project
pip install -r requirements.txt
python app.py
```
Run the app in docker
```bash
# navigate to the project path
cd /path/to/your/project
# When using the docker compose use the below
# since we are using the docker compose we will run the below command
docker-compose up
# stop the container
docker-compose down

# if you are not using docker compose use the below
# build the image
docker build -t handwritten-digit-classifier-web . 
# run the docker
docker run -p 8080:8080 handwritten-digit-classifier-web

# see logs
docker-compose logs web

# stop the container
docker-compose down
# rebuild the container
docker-compose up --build

# check the container
docker ps
```

### Input Data Format
* The data is trained on the MNIST dataset which is a dataset of 28x28 grayscale images of handwritten digits and the labels are the digits themselves.
* The data has black background with white digits
* The data we pass should be a 28x28 grayscale image of a handwritten digit with a black background and white digit
* If you are not passing the data in that format, you can convert the image to that format in the below function 
  * `preprocess_image` in the file `src/handwritten_digit_classifier/utils/common.py`
* The prediction function looks for the image in the below path and processes it
  * `data/image.png`
* The image should be a black background with white digit. We are not doing any image processing to convert the image to this format

### Files are updated in the below order
### Workflow
* **Come of the below files are not needed**. I have added for completeness as a template so this can be referred in other projects 
* Run the project by running the main.py
* We update the below files in that order to achieve the Computer Vision pipeline:
1. config > config.yaml
   1. data_ingestion
   2. data_validation
   3. data_transformation
   4. model_training
2. schema.yaml (**Not Needed for this project**)
   1. The schema of the data i.e. the column headers and the data types
   2. data_validation
3. params.yaml
   1. The hyperparameters for the model
   2. model_training
4. Update the entity
   1. In src > entity > config_entity.py
5. Update the configuration manager 
   1. In src > config > configuration.py
6. Update the components 
   1. In src > components 
      1. data_ingestion.py
      2. data_validation.py
      3. data_transformation.py
      4. model_trainer.py
      5. model_evaluation.py
7. Update the pipeline
    1. In src > pipeline
        1. data_ingestion.py
        2. data_validation.py
        3. data_transformation.py 
        4. model_trainer.py
        5. model_evaluation.py
        6. prediction.py
           1. The prediction pipeline is written only in this file the other above steps are not used in the prediction pipeline
8. Update the main.py
    1. In main.py
