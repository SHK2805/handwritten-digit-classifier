# Handwritten Digit Classifier

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
