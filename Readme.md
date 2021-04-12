# TEXT-2-SQL
A preliminary study on the ability of using ML to translate natural language into SQL
# TEXT TO SQL - Using ML to generate structured queries from Natural Language

Alberto Marengo<br />
marengo.albert@gmail.com<br />
https://github.com/almarengo<br />

---
---


Text to SQL is a preliminary study on the ability of using Machine Learning (ML) to translate natural language into SQL.

**Summary**<br />
The data was parsed and loaded into a pandas dataframe through a customized python class. Preprocessing included cleaning and simplification (class reduction, condition was approximated to its base entity and encoded as number, multiple WHERE statements were eliminated and queries with no WHERE conditions were dropped). The final result of the pre-processing was a datafile for the question (features) and a datafile for the five query components (targets), five multi-class outputs (multi-class multi-output classification). All the five outputs suffered from class imbalance. Using NLP, the questions were vectorized/tokenized in order to have a 2D matrix to feed into ML models.

Three multi-output classifiers were trained:
* Classifier Chain with SVC estimator. The estimator was chosen after exhaustive search of estimators (Logistic Regression, SVC, Random Forest and MultinomialNB) and hyperparameter optimization using GreadSearchCV
* Recurrent Neural Network (RNN) base model
* RNN with class weights to address class imbalance

---
---

Documents in this `project` folder include: 

* 00_Final Report and Presentation files/:
    * 1_Alberto_Marengo_Capstone_final_report - This is the final report that summarizes the `project`
    * 2_Alberto_Marengo_Capstone_Progress_standup - This was the slide deck presented as part of the progress standup
    * 3_Alberto_Marengo_Capstone_Final_Present - This is the slide deck presented as part of the final presentation

* 01_WikiSQL-master/:<br />
    This is the [wikiSQL](https://github.com/salesforce/WikiSQL) repository cloned from `github`. <br />
    Inside this repo the folder `predictions_alberto` was created. Inside are all the file necessary for the evaluation.<br />
    To get the performance of the classifier chain for example, from the main folder run:<br />
    ```python evaluate.py  predictions_alberto/test_true.jsonl predictions_alberto/test.db predictions_alberto/test_chain_pred.jsonl```


Project working files:<br />

* data/:<br />
    Contains the original `.jsonl` files (train, test and validation)

* export_figures/:<br />
    Contains exported figures of the RNN confusion matrices

* figures/:<br />
    Contains usuful figures used in the jupyter notebooks to explain concepts

* load/:<br />
    Contains the script `load_dataset.py` to load the data and reduce classes

* models/:<br />
    Contains the trained and saved models.

* predictions/:<br />
    Contains the encoded predictions of the three models as `.jsonl` files

* processed_data/:<br />
    Contains the processed data split into questions (X) and targets (y) for train, test and validation sets

* The process in Jupyter Notebooks:

    * Alberto_Marengo_Capstone_Notebook_1_Load_Data.ipynb
    * Alberto_Marengo_Capstone_Notebook_2_NLP_EDA.ipynb
    * Alberto_Marengo_Capstone_Notebook_3_Train_model_classification.ipynb
    * Alberto_Marengo_Capstone_Notebook_4_Train_RNN_model_Base.ipynb
    * Alberto_Marengo_Capstone_Notebook_5_Train_RNN_model_Class_Weights.ipynb
    * Alberto_Marengo_Capstone_Notebook_6_model_Evaluation.ipynb

* requirements.txt<br />
    Contains the `capstone` environment with the list of libraries used
