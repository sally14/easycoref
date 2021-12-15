import pandas as pd
import tempfile
import json
import os
import re
import numpy as np
import json
import spacy
import logging;
logging.basicConfig(level=logging.INFO)
import neuralcoref
nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)
import colorama
from colorama import Fore, Back, Style


class CorefModel:
    def __init__(self):
        return None


    def import_dataset(self,path,colnames, filetype='csv'):
        """
        Import the dataset of interest, check if colnames is at the right format and set dataset and colnames attributes
        Args:
            path: string 
                pathfile of the dataset the will be used for coreference detection
            colnames: str or list of str if multiple columns
                columns of the dataset for which we want to predict coreference chain
        Returns:
            df: dataset
        """   
        if filetype=="csv":
            df = pd.read_csv(path)
        elif filetype=="jsonl":
            df = pd.read_json(path, orient='records', lines=True)
        else:
            raise ValueError(f'Type of file {path} is not handled. Filetype must be csv or jsonl')
        self.df = df
        # Check if the columns are at the right format and set attribute colnames
        if type(colnames) == list :
            self.colnames = colnames
        else : 
            if type(colnames)==str:
                self.colnames = [colnames]  
            else :
                print('Argument colnames is not a list of string or a string')
                raise TypeError      

        return self.df


    def clean(self):
        """
        Requires to have run the method import_dataset before
        Check if the columns of interest are strings and prepocess the columns
        Returns:
            df: dataset
        """   
        for col in self.colnames :
            # Check if columns are strings 
            if self.df.dtypes[col] == str :
                # Replace wrong typos
                self.df[col] = self.df[col].str.replace('\n','. ')
                self.df[col] = self.df[col].str.replace('  ',' ')
            elif self.df.dtypes[col] == object :
                self.df[col] = self.df[col].astype(str)
                self.df[col] = self.df[col].str.replace('\n','. ')
                self.df[col] = self.df[col].str.replace('  ',' ')
            else :
                print(f"Column type {col} is not string")
                raise TypeError
            return self.df
    

    def __transform_neuralcoref__(self):
        """
        Requires to have run the method import_dataset and clean before
        Set the dataset to an adapted form for evaluation using NeuralCoref model
        Returns: 
            df_eval: dataset 
        """ 
        # NeuralCoref only needs dataset with the columns of interest
        self.df_eval = self.df
        return self.df_eval




    def __split_into_sentences__(self, text):
        """ 
        -- Sub-function : step of __transform_e2ecoref__ --
        Split a text into a list of sentences
        Args: 
            text : str
        Returns:
            sentences : list of str 
                list of the text sentences
        """

        # Typos needed
        alphabets= "([A-Za-z])"
        prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
        suffixes = "(Inc|Ltd|Jr|Sr|Co)"
        starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
        acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
        websites = "[.](com|net|org|io|gov)"

        # Fixe the typos
        text = " " + text + "  "
        text = text.replace("\n"," ")
        text = re.sub(prefixes,"\\1<prd>",text)
        text = re.sub(websites,"<prd>\\1",text)
        if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
        text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
        text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
        text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
        text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
        text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
        if "”" in text: text = text.replace(".”","”.")
        if "\"" in text: text = text.replace(".\"","\".")
        if "!" in text: text = text.replace("!\"","\"!")
        if "?" in text: text = text.replace("?\"","\"?")
        text = text.replace(".",".<stop>")
        text = text.replace("?","?<stop>")
        text = text.replace("!","!<stop>")
        text = text.replace("<prd>",".")

        # Split into sentences
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]

        return sentences


    def __formatage_liste__(self, text):
        """ 
        -- Sub-function : step of __transform_e2ecoref__ --
        Split the text into a list of list of words for each sentence and associate it a list of list of speakers for each word
        
        Args: 
            text : str
        Returns:
            [liste_formate, liste_speaker] : list of two lists
                liste_formate: list of lists 
                    list of words for each sentence of a text
                liste_speaker: list of lists 
                    speaker for each word of liste_formate (empty strings by default)
        """
        liste_formate = []
        liste_speaker = []

        # Create a list of sentences
        liste_sentence = self.__split_into_sentences__(text)

        # Case only one sentence
        if liste_sentence == [] :
          liste_formate = [list(text.split(" "))]
          liste_speaker = ["" for i in liste_formate]

        else :
            # Transform each sentence to a list of words and create a list of empty strings
            for sentence in liste_sentence :
                liste_mot = list(sentence.split(" ")) 
                liste_speak = ["" for i in liste_mot]

                # Add them to the "global" list
                liste_formate.append(liste_mot)
                liste_speaker.append(liste_speak)

        return [liste_formate, liste_speaker]


    def __dico__(self, text):
        """ 
        -- Sub-function : step of __transform_e2ecoref__ --
        Create a dictionnary for a text under a specific format, needed to use e2ecoref

        Args: 
            text : str
        Returns:
            dico : dictionnary
                dictionnary for the text under the following format :
                    - clusters : where the coreference chains of the text will be added
                    - doc_key : genre of the text, set by default to "nw" news wire, but can be changed ("bc": broadcast conversations, 
                    "bn": broadcast news, "mz": magazines, "nw": news wire, "pt": pivot corpus, "tc": telephone conversation, "wb": weblogs)
                    - sentences : list of the text sentences, each being under a list of words format
                    - speakers : speaker for each word, respecting the same format as previous "sentences" 

        """
        
        dico={
        "clusters": [],
        "doc_key": "nw",
        "sentences": self.__formatage_liste__(text)[0],
        "speakers": self.__formatage_liste__(text)[1]
        }

        return dico




    def __transform_e2ecoref__(self,col):
        """
        Requires to have run the method import_dataset and clean before
        For one specific column, create a json file to an adapted form for inference with e2eCoref model. Can be used successively 
        if there are several column of interest for one dataset.
        Args:
            col : str
                name of the specific column for which we want to use the model
        Returns: 
            df_eval : dataset 
                dataset of the json file with the following columns :
                    - clusters : where the coreference chains of the text will be added
                    - doc_key : genre of the text, set by default to "nw" news wire, but can be changed ("bc": broadcast conversations, 
                    "bn": broadcast news, "mz": magazines, "nw": news wire, "pt": pivot corpus, "tc": telephone conversation, "wb": weblogs)
                    - sentences : list of the text sentences, each being under a list of words format
                    - speakers : speaker for each word, respecting the same format as previous "sentences" 

        """  

        # Create the dictionnary  
        dicos_list = []

        # For each text of the column create a dico under the right format
        for text in self.df[col]:
            dicos_list.append(self.__dico__(text))

        # Saving the dataset under a json temporary file 
        self.fp1 = tempfile.NamedTemporaryFile(mode='ab')
        datapath = self.fp1.name
  
        # For each line of the dictionnary
        for dico in dicos_list :
            self.fp1.write(bytes(json.dumps(dico), 'utf-8'))
            self.fp1.write(b'\n')
        
        # Read the saved json temporary file 
        df_eval = pd.read_json(datapath, orient='records', lines=True, encoding='utf-8')
        
        return df_eval




    def __neuralcoref__(self,col):
        """
        Requires to have run the method import_dataset, clean and __transform_neuralcoref__ before
        Gives the coreference chain clusters of each text for column col of the dataset, using the NeuralCoref model 

        Args:
            col : str
                name of the specific column for which we want to use the model
        Returns: 
            column_coref : list of list of lists of spans
                future dataset column, each element i being a list of every coreference clusters 
                found by the model for text line i of the dataset. A coreference cluster is a list of 
                text "spans" (specific class used by NeuralCoref model)
        """ 

        column_coref = []
        # For each text of the dataset
        for i in range(len(self.df_eval)):
            text = self.df_eval[col][i]
            text_nlp = nlp(text)
            # Use neuralcoref module to give the coreference chains clusters of the text
            column_coref.append(text_nlp._.coref_clusters)

        return column_coref
            
 
    def __e2ecoref__(self,col):
        """
        Requires to have run the method import_dataset, clean and __transform_e2ecoref__ before
        Gives the dataframe presenting results of coreference detection using e2eCoref model for one specific column of text.
        Args: 
            col : str 
                name of the specific column for which we want to use the model
        Returns: 
            df_coref[col] : dataframe 
                dataframe with the following columns :
                    (each column line corresponding to the evaluation of the text of the dataframe line i)
                    - clusters : empty lists
                    - doc key : "nw" by default (news wire)
                    - sentences : list of lists, each sentences of the text splitted into list of words
                    - speakers : list of lists, gives speakers for each word, following previous "sentences" format
                    - predicted_clusters : list of lists, for each clusters of coreference chain found gives list of mention positions 
        """
        os.chdir('./e2e-coref')
        # Call the temporary file used for method __transform_e2ecoref__ (which create the file used for evaluation)
        datapath = self.fp1.name 
        # Create new temporary file for evaluation output
        self.fp2 = tempfile.NamedTemporaryFile(mode='ab')
        output = self.fp2.name

        # Prediction using e2eCoref
        os.system(f'python ./predict.py final {datapath} {output}')
        self.fp1.close()

        df_coref = pd.read_json(output, orient='records', lines=True, encoding='utf-8')
        setattr(self, f'df_coref_{col}', df_coref)
        self.fp2.close()
        os.chdir('..')
        return getattr(self, f'df_coref_{col}')


    def inference(self,model):
        """
        Requires to have run the method import_dataset and clean before
        Arguments : self, model (NeuralCoref or e2eCoref)
        Detect and extract coreference chains for each text and column of the dataframe and present the results in a new dataframe.
        
        If model is e2ecoref, also create a dataframe used later for standardized results :
        df_useful : dataframe
             for each column col : 
                - text_list_col : text under list format
                - predicted_clusters_col : coreference mentions positions under list format
        
        Args: 
            model : str 
                name of the model we want to use for coreference detection (neuralcoref or e2ecoref)
        Returns: 
            df_results : dataframe 

            if model is neuralcoref :
                dataframe with the following columns, for each column col :
                    (each column line corresponding to the evaluation of the text of the dataframe line i)
                    - col : texts of the original dataframe
                    - clusters_col : list of lists of spans, for each clusters of coreference chain gives list of spans
                    (this specific class used by NeuralCoref contains in itself the span start and end positions) 

            if model is e2coref :
                dataframe with the following columns, for each column col :
                    (each column line corresponding to the evaluation of the text of the dataframe line i)
                    - col : texts of the original dataframe
                    - clusters_col : list of lists of strings, for each clusters of coreference chain gives list of mentions
        """

        # Create dataframe for results 
        self.df_results = pd.DataFrame()

        # According to the model given in argument, create columns of results

        if model == "neuralcoref":
            # Run the method required before using the methode __neuralcoref__
            self.df_eval = self.__transform_neuralcoref__()

            for col in self.colnames :
                # Column of col texts
                self.df_results[col] = self.df[col]
                # Column of clusters found for each col
                self.df_results[f'clusters_{col}'] = self.__neuralcoref__(col)

        elif model == "e2ecoref" :
            # Create dataframe for further use 
            self.df_useful = pd.DataFrame()

            # For each column of colnames
            for col in self.colnames :
                # Run methods required 
                df_eval = self.__transform_e2ecoref__(col)
                df_coref = self.__e2ecoref__(col)

                # Column of col texts
                self.df_results[col] = self.df[col]
            
                # Give the mention string of the mentions positions given by column predicted_clusters
                column_cluster = []
                column_text_list = []

                for i in range(len(self.df_results)):
                    liste_clusters = []
                    list_clusters_num = df_coref["predicted_clusters"][i]
                    list_sentences = df_coref['sentences'][i]
                    # Text under list format
                    flat_list_sentences = [item for sublist in list_sentences for item in sublist]

                    for cluster in list_clusters_num :
                        # List of the cluster strings
                        cluster_str = []
                        # For each mention of the coreference chain
                        for item in cluster :
                            mention_start = item[0]
                            mention_end = item[1] + 1
                            # Add the mention string to the cluster of strings
                            cluster_str.append(flat_list_sentences[mention_start:mention_end])

                        # Add the cluster of strings to the list of clusters
                        liste_clusters.append(cluster_str)

                    # Add the list of clusters of text line i to the column
                    column_cluster.append(liste_clusters)
                    # Add the text under list format to the column
                    column_text_list.append(flat_list_sentences)

                # Column of col texts
                self.df_results[f'clusters_{col}'] = column_cluster

                # Column of text_list add to useful dataframe
                self.df_useful[f'text_list_{col}'] = column_text_list
                # Column of predicted_clusters (mentions positions under list format) add to useful dataframe
                self.df_useful[f'predicted_clusters_{col}'] = df_coref["predicted_clusters"]
                
        else:
            print('This model is not manageable with CorefModel')
            raise NameError
        
        return self.df_results







    def __isprefixe__(self,i,mot,texte): 
        """ 
         -- Sub-function : step of standardization and visualisation --
        Check if a word has an occurrence in a text in position i
        Args: 
            i : int
            mot : str
            texte : str
        Returns: 
            B : bool
        """

        B = True
        j=0
        while (j < len(mot)) and B:
            if texte[i+j] != mot[j]:
                B = False
            j+= 1 
        return B


    def __positions_str__(self,mention_str,texte): 
        """ 
        -- Sub-function : step of standardization and visualisation --
        Give list of occurring positions of a mention in a text
        Args: 
            mention_str : str
            texte : str 
        Returns: 
            occ : list 
                list of the occurring positions of the mention in the text
        """
        occ = []
        for i in range(len(texte)-len(mention_str)+1):
            if self.__isprefixe__(i,mention_str,texte): 
                occ.append(i)

        return occ


    def __positions_span__(self, mention_str,texte): 
        """ 
        -- Sub-function : step of standardization and visualisation --
        Give list of occurring start span positions of a mention in a text
        Args:
            mention_str : str
            texte : str
        Returns:
            occ1 : list 
                list of the start span positions of the mention in the text
        """
        occ1 = []
        for i in self.__positions_str__(mention_str,texte): 
            
            chaine = texte[0:i+len(mention_str)]
            mention_span = nlp(mention_str)
            chain = nlp(chaine)

            occ1.append(len(chain)-len(mention_span))
    
        return occ1

    def __positions_list__(self,mention,texte):
        """
        -- Sub-function : step of standardization and visualisation --
        Give list of start and end positions of a mention in a text
        Args:
            mention : str 
            texte : str
        Returns: 
            occ2 : list of lists ([start,end] format)
                list of the start and end list positions of the mention in the text
        """

        occ2 = []
        for i in self.__positions_str__(mention,texte): 
            
            # To handle text part without any complete sentence
            chaine = texte[:i] +  "."

            liste = self.__formatage_liste__(chaine)[0]

            # Transform list of lists to a simple list
            liste_flat = [item for sublist in liste for item in sublist]

            # Handle empty list
            if liste_flat == [] :
                liste_flat = chaine.split(" ")
            liste_flat.pop()
           
            mention_list = mention.split(" ")
            position = len(liste_flat) 
            occ2.append([position,position+len(mention_list)])
 
        return occ2


    def __position_span_to_str__(self,mention,texte): 
        """ 
        -- Sub-function : step of standardization and visualisation --
        Give the start string position of a mention in a text based on its span positions
        Args: 
            mention : span
            texte : str 
        Returns: 
            position_finale : int
                start string position of the mention in the text
        """

        mention_str = mention.text

        span_position = mention.start 

        # Function returning the list of string positions of the mention in the text
        liste_pos_str = self.__positions_str__(mention_str,texte) 
        # Function returning the list of span positions of the mention in the text
        liste_pos_span = self.__positions_span__(mention_str,texte) 
        
        # Check if the span position of the mention is in the list of span positions of the mention in the text
        if span_position in liste_pos_span :
            # Take the list index of that span positions
            ind = liste_pos_span.index(span_position)
            # Take the parallel string position corresponding to that index
            position_finale = liste_pos_str[ind]
    
        return position_finale 

    def __position_str_to_span__(self,start,end,texte): 
        """ 
        -- Sub-function : step of standardization and visualisation --
        Give start and end span positions of a mention in a text based on its string positions
        Args: 
            start : int
                start str position of a mention
            end : int
                end str position of a mention
            texte : str 
        Returns: 
            list ([start, end] format)
                start and end span positions of the mention in the text
        """
        mention_str = texte[start:end]
        mention_span = nlp(mention_str)

        chaine = texte[0:end]
        chain = nlp(chaine)

        return ([len(chain)-len(mention_span),len(chain)])

    def __position_list_to_str__(self,position,mention,texte): 
        """ 
        -- Sub-function : step of standardization and visualisation --
        Give start str position of a mention in a text based on its start list position
        Args: 
            position : int
                start position of the mention under list format
            mention : str
            texte : str
        Returns: 
            position_finale : int
                start string position of the mention in the text
        """
        # Function returning the list of string positions of the mention in the text
        liste_pos_str = self.__positions_str__(mention,texte)
        # Function returning the list of list positions of the mention in the text
        liste_pos_list = self.__positions_list__(mention,texte)

        # Check if the list position of the mention is in the list of list positions of the mention in the text
        if position in liste_pos_list :
            # Take the list index of that list positions
            ind = liste_pos_list.index(position)
            # Take the parallel string position corresponding to that index
            position_finale = liste_pos_str[ind]
      
        return position_finale 



    def __no_doublons__(self,clusters):
        """ 
        -- Sub-function : step of visualisation --
        Find every overlapping mentions of detected coreference chains for a text, 
        and only keep the one with the best coreference score
        Args: 
            clusters : list of lists of spans
                all clusters of coreference chain found by neuralcoref for a particular text
        Returns: list
                list of mentions to supress because they overlaps others 
        """
        liste_positions = []
        liste_mentions = []
        liste_mentions_a_suppr = []
        for clust in clusters :
            cluster = clust.mentions
            
        
            for mention in cluster:
                # List of all intervall of spans
                liste_positions.append(pd.Interval(mention.start, mention.end))
                # List of all spans
                liste_mentions.append(mention) 

 
        # Observe if some overlaps each others
        for interval1 in liste_positions :
            for interval2 in liste_positions :
                if interval1.overlaps(interval2) and interval1 != interval2 :

                    i1 = liste_positions.index(interval1) 
                    i2 = liste_positions.index(interval2)
                    mention1 = liste_mentions[i1]
                    mention2 = liste_mentions[i2]


                    dico1 = mention1._.coref_scores
                    score1 = max(dico1.values())

                    dico2 = mention2._.coref_scores
                    score2 = max(dico2.values())
                
                    # Add the mention with the lower score to the list of mention to suppress
                    if score1 <= score2 and [mention1.start,mention1.end] not in liste_mentions_a_suppr :
                        liste_mentions_a_suppr.append([mention1.start, mention1.end])
                    
                    elif score1 > score2 and [mention2.start,mention2.end] not in liste_mentions_a_suppr :
                        liste_mentions_a_suppr.append([mention2.start, mention2.end])
                    

        return(liste_mentions_a_suppr)



    def __standardized_results__(self,model):
        """
        Requires to have run the method import_dataset, clean and inference before
        Gives a dataframe of standardized results that will be useful for visualization
        Args: 
            model : str 
                name of the model we want to use for coreference detection (neuralcoref or e2ecoref)     
        Returns: 
            df_standardized : dataframe 
                dataframe with the following columns, for each column col  :
                    (each column line corresponding to the evaluation of the text of the dataframe line i)
                    - col : texts of the original dataframe 
                    - clusters_col : list of lists, for each clusters of coreference chain gives list of mentions 
                    (if model is neuralcoref mentions are spans and if model is e2ecoref mentions are strings) 
                    - span_positions_col : list of list of lists (format [index_start,index_end]), for each clusters of coreference 
                    chain gives list of mention position under span format
        """

        if model == "neuralcoref":
            # Dataframe with columns col and clusters_col
            self.df_standardized = self.df_results

            # Build columns span_positions_col 
            for col in self.colnames :

                # Create column giving the span positions of the mentions of coreference chains of each text
                column_span_pos = []
                for i in range(len(self.df_standardized)) :
                    # List of lists : span positions of every mention of every cluster for one text
                    text_span_pos = []

                    # Mentions to suppress
                    mentions_a_supp = self.__no_doublons__(self.df_standardized[f'clusters_{col}'][i])
                    

                    for clusters in self.df_standardized[f'clusters_{col}'][i]:
                        cluster = clusters.mentions
                        cluster = [mention for mention in cluster if [mention.start,mention.end] not in mentions_a_supp]
                        
                
                        # List of span positions of every mention of one cluster
                        cluster_span_pos = []
                        for mention in cluster :
                            # Mention are spans : add start and end span position to the cluster 
                            cluster_span_pos.append([mention.start, mention.end])

                        # Add the cluster list of mention positions for each cluster 
                        text_span_pos.append(cluster_span_pos)
                    
                    # Add the list of lists for each text to the column
                    column_span_pos.append(text_span_pos)

                self.df_standardized[f'span_positions_{col}'] = column_span_pos


        elif model == "e2ecoref" :
          # Dataframe with columns col and clusters_col
          self.df_standardized = self.df_results

          # Build columns span_positions_col 
          for col in self.colnames:

            # Create column giving the string positions of the mentions of coreference chains of each text
            column_str_pos = []

            for i in range(len(self.df_useful)):
              # List of lists : string positions of every mention of every cluster for one text
              text_str_pos = []

              for cluster in self.df_useful[f'predicted_clusters_{col}'][i] :
                
                # List of string positions of every mention of one cluster
                cluster_str_pos = []

                # Convert list positions to string positions
                for positions in cluster:
               
                  start = positions[0]
                  end = positions[1]+1
                  positions_corr = [start,end]
                  
                  mention = self.df_useful[f'text_list_{col}'][i][start:end]
                  mention_str = " ".join(mention)
            
                              
                  texte = self.df_standardized[col][i]
                  
                                      
                  # Start string position of the mention
                  pos_fin = self.__position_list_to_str__(positions_corr,mention_str,texte)

                  # Add start and end string positions to the list 
                  cluster_str_pos.append([pos_fin,pos_fin+len(mention_str)])

                # Add the cluster list of mention positions for each cluster 
                text_str_pos.append(cluster_str_pos)

              # Add the list of lists for each text to the column
              column_str_pos.append(text_str_pos)

            self.df_useful[f'str_pos_{col}'] = column_str_pos
        

            # Thanks to that new column, create column giving the spans positions of the mentions of coreference chains of each text
            column_span_pos = []
            for i in range(len(self.df_useful)):
              text_span_pos = []
              for cluster in self.df_useful[f'str_pos_{col}'][i] :
                cluster_span_pos = []
                for pos_str in cluster :
                  start = pos_str[0]
                  end = pos_str[1]
                  texte = self.df_standardized[col][i]
                  pos = self.__position_str_to_span__(start,end,texte)
                    
                  cluster_span_pos.append(pos)
                    
                # Add the cluster list of mention positions for each cluster 
                text_span_pos.append(cluster_span_pos)

              # Add the list of lists for each text to the column  
              column_span_pos.append(text_span_pos)

            self.df_standardized[f'span_positions_{col}']= column_span_pos
            
        else:
            print('This model is not manageable with CorefModel')
            raise NameError

        
        return self.df_standardized



    def visualisation(self,model,col,i):
        """ 
        Requires to have run the method import_dataset, clean and inference before. 
        Given a text, highlights all the coreference chains detected by the chosen model. 
        This function must be printed to see the highlights in different colors.
        Args: 
            model : str
                name of the model we want to use for coreference detection (neuralcoref or e2ecoref)
            col : str
                column of interest (in colnames)
            i : int
                line of interest (in the column)
        Returns:  
            texte : str
                text with all of it coreference chains underlined in different colors
        """
        if model not in ['neuralcoref', 'e2ecoref']:
          print('This model is not manageable with CorefModel')
          raise NameError
        
        else:
          self.df_standardized = self.__standardized_results__(model)
          texte = self.df_standardized[col][i]
          texte_or = texte 
          nlp_texte = nlp(texte)
                
          liste_charactere = [i for i in range(len(texte))]
          liste_charactere_updated = [i for i in range(len(texte))]

            # Font color
          color = 0 
          colors = 240

          clusters_positions = self.df_standardized[f'span_positions_{col}'][i]
            
          for cluster in clusters_positions :
            color += 1
            
            if len(cluster)>1 :
              for positions in cluster :

                # Positions in spans
                mention_start = positions[0]
                mention_end = positions[1]

                # Mention in span
                mention = nlp_texte[mention_start:mention_end]
                # Mention in str
                mention_str = (nlp_texte[mention_start:mention_end]).text 

                # Mention start position in strings
                index_position_start = self.__position_span_to_str__(mention,texte_or) 
                position_start = liste_charactere_updated[index_position_start]
                # Mention end position in strings
                position_end = position_start+len(mention_str) 

                # Text from beginning to mention
                deb = texte[0: position_start] 
                # End of the text
                fin = texte[position_end:] 

                # Rewrite the text
                texte = deb + f'\033[38;5;{color}m' + f'\x1b[48;5;{colors}m' + mention_str + '\033[0;0m' + fin #on modifie texte en changeant la couleur de la mention
                add1 = len(f'\033[38;5;{color}m') + len(f'\x1b[48;5;{colors}m')
                add2 = len('\033[0;0m')


                # Update positions of text element after adding add1
                for i in range(index_position_start,len(liste_charactere_updated)): 
                  liste_charactere_updated[i] += add1
                # Update positions of text element after adding add2
                for i in range(index_position_start+len(mention_str),len(liste_charactere_updated)): 
                  liste_charactere_updated[i] += add2
                
                
                
        return texte

    
 
