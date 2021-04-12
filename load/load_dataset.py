
import json
import spacy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

class CreateData():

    """
    A class to load the wikiSQL dataset from the provided .jsonl file 
    and create a dataframe that has the following columns:
    table_id, question, select_col, agg_col, first_where_col_0, first_where_col_1, first_where_col_2, second_where_col_0, second_where_col_1, second_where_col_2 

    ...

    Attributes
    ----------
    df : pandas DataFrame
        the dataframe created from the source file
    entity_list : dict
        the encoding of the entities of the last WHERE column
    
    Methods
    -------
    plot():
        Plots the class distributions of the five output columns

    reduce_return_x_y(self, keep_list=None, keep=False):
        Reduces the classes according to passed arguments and returns the X datafile (question) and y datafile (five target columns).
    """

    def __init__(self, path_file):

        """
        Constructs the dataframe. 

        Parameters
        ----------
            path_file : str
                the location of the .jsonl file
            
        """
        # Check that a path_file is passed and that it's a valid string
        assert path_file is not None, 'Must enter a valid path and file name.'
        assert isinstance(path_file, str), 'Path and file name must be a string.'

        # Open json and load the data into a dataframe
        data = []
        with open(path_file) as f:
            for line in f:
                data.append(json.loads(line))
        df = pd.DataFrame(data)
        # Unpack the sql dictionary column
        # Assign select_col to values in key 'sel'
        df['select_col'] = df.sql.apply(lambda x: x['sel'])
        # Assign agg_col to values in key 'agg'
        df['agg_col'] = df.sql.apply(lambda x: x['agg'])
        # Assign where_col to values in key 'conds'
        df['where_col'] = df.sql.apply(lambda x: x['conds'])
        # Explode where_col column from the list of lists, create multiple rows for queries with multiple WHERE statements
        df.explode('where_col')
        # Create a where_cond dataframe: 
        # wher_col_0 column contains the list of the first WHERE or NaN if no condition
        # wher_col_1 column contains the list of the second WHERE or NaN if no second condition
        where_conds = df['where_col'].apply(pd.Series)
        where_conds = where_conds.rename(columns = lambda x: 'where_col_'+str(x))
        # Create a where_cond_0 dataframe: each column is an element of the list of first WHERE condition 
        where_conds_0 = where_conds['where_col_0'].apply(pd.Series)
        where_conds_0 = where_conds_0.rename(columns = lambda x: 'first_where_col_'+str(x))
        # Create a where_cond_1 dataframe: each column is an element of the list of second WHERE condition 
        where_conds_1 = where_conds['where_col_1'].apply(pd.Series)
        where_conds_1 = where_conds_1.rename(columns = lambda x: 'second_where_col_'+str(x))

        # Stack df and the unpacked newly created where dataframe with unpacked elements
        df = pd.concat([df, where_conds_0, where_conds_1], axis=1).reset_index(drop=True)
        # Call method .encode_where_col on df to encode the entities in the last WHERE column to numeric
        entities = self.encode_where_col(df)
        # Insert the entities in df as a new column in position 10
        df.insert(loc=10, column='first_where_col_2_ent', value=entities)
        
        # Try if the entiy_conversion file exists, meaning that this is not the first time that a dataframe has been created
        # i.e. we are creating the validation or the test set. In this case we want to use the same encoding we used for the train set
        try:
            with open('processed_data/entity_conversion.json') as f:
                temp_dict = json.load(f)
                map_dict = {}
                # Create an ecoding dictionary from the json file: entities are keys and encoding numbers are values
                for i, val in temp_dict.items():
                    map_dict[val] = i
            self.entity_list = temp_dict
        except:
            # If entity_conversion does not exist (first time encoding) create the encoding dictionary
            # Create an ordered list of most frequent entities
            values = df['first_where_col_2_ent'].value_counts().index
            map_dict = {}
            # Create an ecoding dictionary: entities are keys and encoding numbers (from enumerate) are values
            for i, val in enumerate(values):
                map_dict[val] = i
            self.entity_list = dict(enumerate(values))
        
        # Create a column that encode the enties to their numbers
        df['first_where_col_2'] = df['first_where_col_2_ent'].map(map_dict)
        # Drop columns not needed
        df.drop(['phase', 'sql', 'where_col', 'first_where_col_2_ent'], axis=1, inplace=True)
        self.df = df
        
        # Print a message the dataframe is created and a brief descriptions of options
        print(f"Loaded dataset shape: {self.df.shape}")
        return print("Instance created!! Database is loaded! \n Call .df to see it. \n Call .entity_list to retrieve the entity converter list. \n Call .plot() for class distribution! \n Call .reduce_return_x_y() to make X and y!")
    
    def plot(self, num_feat=5):

        """
        Plots the class distribution in the target columns.

        Parameters
        ----------
            num_feat=5 : int
                number of targets. Default is 5 (select_col, agg_col, first_where_col_0, first_where_col_1, first_where_col_2).
            
        """
        # Create the figure and axes
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
        # Flatten axes a 1D vector
        ax = ax.flatten()

        # Loop through the num of targets
        for i in range(num_feat):
            # Get the names of the most frequent values
            values = self.df.iloc[:, i+2].value_counts().index.astype(str)
            # Get the most frequent values
            counts = self.df.iloc[:, i+2].value_counts(normalize=True)
            # Get the column name for labels and title
            col_name = self.df.iloc[:, i+2].name
            # Plot th bar chart
            ax[i].bar(values, counts*100)
            ax[i].set_xlabel(col_name)
            ax[i].set_ylabel('% of questions')
            sns.despine()
            ax[i].set_title(f"{col_name} column distribution")

        # Title of the overall figure
        fig.suptitle("Query columns distributions", size = 20)
        fig.tight_layout()

        # Delete the last subplot
        fig.delaxes(ax[-1])
    
    def encode_where_col(self, df):

        """
        Internal function that get the entities in the question column of df and return a list of entities (one per row).

        """
        # Load the module Entity Recognition from spaCy
        nlp = spacy.load("en_core_web_sm")
        entities = []
        doc_ents = ''
        # Loop over the questions in df
        for i in range(len(df['question'])):
            # Extract the entity in the question
            doc = nlp(df['question'].iloc[i])
            # Try if the question contains an entity
            try: 
                # Store it in a string
                doc_ent = doc.ents[0]
            # If there is no entity, avoid error and append word "empty" in entity list
            except:
                entities.append("empty")
            # If there is an entity, append the entity to entity list
            else:
                entities.append(doc_ent.label_)
        return entities


    def reduce_classes(self, keep_list):

        """
        Internal function that reduces the number of classes based on a list of number of class to keep.

        """

        # Loop over the length of the list (number of targets)
        for i in range(len(keep_list)):
            # If the positional argumant is 'all' --> all the classes are kept 
            if keep_list[i] == 'all':
                # Continue to loop over the other arguments
                continue
            # Get the ordered frequencies of each class
            counts = self.df.iloc[:, i+2].value_counts()
            # If last WHERE column
            if i == (len(keep_list) - 1):
                # Try to load the entity_train.csv file, meaning that this is not the first time that a data reduction has been done
                # i.e. we are creating the validation or the test set. In this case we want to keep the same classes we kept for the train set
                try:
                    counts = pd.read_csv('processed_data/entities_train.csv', index_col=0)
                    counts.index = counts.index.astype(str)
                # If error save the ordered frequencies of this class into the entity_train.csv file
                except:
                    counts.to_csv('processed_data/entities_train.csv')
            map_dict = {}
            # Loop over range of the number of classes to keep
            for j in range(keep_list[i]-1):
                # Create a dictionary where key=value for mapping
                map_dict[counts.index[j]] = counts.index[j]
            # Assign at each class the class value only if in the top keep_list[i], NaN to all the others
            self.df.iloc[:, i+2] = self.df.iloc[:, i+2].map(map_dict)
            # Fill NaN values with next available index
            self.df.iloc[:, i+2].fillna(counts.index[j+1], inplace=True)


    def reduce_return_x_y(self, keep_list=None, keep=False):

        """
        Returns the X datafile (question) and y datafile (five target columns), inputs for ML models.

        If the argument 'keep_list' is passed, then the classes are reduced according to the arguments in list.

        Parameters
        ----------
        keep_list : int and/or str
            A list of how many class to keep. List must contain 5 arguments (one per traget). 
            List can contain integer numbers or string 'all' to keep all the classes for the positional argument.

        keep: boolean
            Default is False. If no class reduction is needed then pass True.

        Returns
        -------
        X: Pandas Series
            Each row is a question
        y: Pandas DataFrame
            The targets of the classification
        """

        # If no list is passes switch keep to True, keeps all the classes
        if keep_list == None:
            keep = True

        # If class reduction is needed
        if not keep:
            # Check a list is passed
            assert isinstance(keep_list, list), "Number of columns to keep need to be a list type"
            # Check the list has five arguments
            assert len(keep_list) == 5, "5 elements need to be inserted in the list, one for each column of the dataframe. Type 'all' to keep all the classes in that column"
            # Apply class reduction with internal function
            self.reduce_classes(keep_list)   

        # Drop second WHERE columns
        self.df.drop(['second_where_col_0', 'second_where_col_1', 'second_where_col_2'], axis=1, inplace=True)
        # Drop rows where there is no WHERE statement
        self.df.dropna(inplace=True) 
        # Create df mask to cast columns to integer
        df_to_numeric = self.df[['select_col', 'agg_col', 'first_where_col_0', 'first_where_col_1', 'first_where_col_2']]
        # Cast the target columns to integer, necessary for classification
        self.df[['select_col', 'agg_col', 'first_where_col_0', 'first_where_col_1', 'first_where_col_2']] = df_to_numeric.astype('int')

        # Create X
        X = self.df['question'].reset_index(drop=True)
        # Create y
        y = self.df[['select_col', 'agg_col', 'first_where_col_0', 'first_where_col_1', 'first_where_col_2']].reset_index(drop=True)

        # Print shapes of X and y
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        return X, y







        
