# Handwritten Digit Classifier


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
