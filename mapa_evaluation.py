import csv
from itertools import count
import pandas as pd
import argparse
from collections import defaultdict
import glob 
import sequence_labelling

def load_TSV_to_DF(directory):

    """ 
        Crawl given directory recursively and load all TSV files to a single dataframe (one global dataframe)
    """

    # print("File loading: load_TSV_to_DF......................................................")

    files = glob.iglob(directory + '**/*.tsv', recursive=True)
    dfs = [pd.read_csv(f, sep='\t', skiprows=1, quoting=csv.QUOTE_NONE, usecols=['TOKEN','LEVEL1_GOLD','LEVEL2_GOLD','LEVEL1_PRED','LEVEL2_PRED','TOKEN_SPAN']) for f in files]

    df = pd.concat(dfs,ignore_index=True)

    return df

def load_each_file(directory):

    """ 
        Load each file of a given directory to a dataframe (one file per dataframe) (?)
    """

    files = glob.iglob(directory + "**/*.tsv", recursive=True)

    dfs = [pd.read_csv(f, sep='\t', skiprows=1, quoting=csv.QUOTE_NONE, usecols=['TOKEN','LEVEL1_GOLD','LEVEL2_GOLD','LEVEL1_PRED','LEVEL2_PRED','TOKEN_SPAN']) for f in files]
    
    # df = pd.read_csv(file, sep='\t', skiprows=1, quoting=csv.QUOTE_NONE, usecols=['TOKEN','LEVEL1_GOLD','LEVEL2_GOLD','LEVEL1_PRED','LEVEL2_PRED','TOKEN_SPAN'])

    # print(dfs)

    return dfs

def get_category(row):
    
    """
        Return categories (PERSON, DATE, family name, etc.) from columns LEVEL1_GOLD, LEVEL2_GOLD, LEVEL1_PRED, LEVEL2_PRED
        If tag is 'O', returns 'O'
    """
    
    gold_1 = row['LEVEL1_GOLD'].split("-", 1)
    pred_1 = row['LEVEL1_PRED'].split("-", 1)

    gold_2 = row['LEVEL2_GOLD'].split("-", 1)
    pred_2 = row['LEVEL2_PRED'].split("-", 1)
                    
    if len(gold_1) > 1: 
        gold_class1 = gold_1[1]
    else:
        gold_class1 = gold_1[0]
        
    if len(pred_1) > 1:
        pred_class1 = pred_1[1]
    else:
        pred_class1 = pred_1[0]
        
    if len(gold_2) > 1: 
        gold_class2 = gold_2[1]
    else:
        gold_class2 = gold_2[0]
        
    if len(pred_2) > 1:
        pred_class2 = pred_2[1]
    else:
        pred_class2 = pred_2[0]   

    # if gold_class1 != pred_class1:
    #     print("level 1", gold_class1, pred_class1)

    # if gold_class2 != pred_class2:
    #     print("level 2", gold_class2, pred_class2)

    return gold_class1, pred_class1, gold_class2, pred_class2

def get_category2(row, level):
    
    """
        NOT IN USE
        Return categories from columns LEVEL1_GOLD, LEVEL2_GOLD, LEVEL1_PRED, LEVEL2_PRED
        If row is 'O', returns 'O'
    """
    
    if level == "1":
        gold_column = 'LEVEL1_GOLD'
        pred_column = 'LEVEL1_PRED'
    elif level == "2":
        gold_column = 'LEVEL2_GOLD'
        pred_column = 'LEVEL2_PRED'

    gold = row[gold_column].split("-", 1)
    pred = row[pred_column].split("-", 1)
                   
    if len(gold) > 1: 
        gold_class = gold[1]
    else:
        gold_class = gold[0]
        
    if len(pred) > 1:
        pred_class = pred[1]
    else:
        pred_class = pred[0] 

    return gold_class, pred_class

def get_boundary(row):
    
    """
        Return boundaries (B or I) from columns LEVEL1_GOLD, LEVEL2_GOLD, LEVEL1_PRED, LEVEL2_PRED
        If tag is 'O', returns O
    """
    
    gold_1 = row['LEVEL1_GOLD'].split("-", 1)
    pred_1 = row['LEVEL1_PRED'].split("-", 1)

    gold_2 = row['LEVEL2_GOLD'].split("-", 1)
    pred_2 = row['LEVEL2_PRED'].split("-", 1)
    
    gold_boundary1 = gold_1[0]
    pred_boundary1 = pred_1[0]
    gold_boundary2 = gold_2[0]
    pred_boundary2 = pred_2[0]

    return gold_boundary1, pred_boundary1, gold_boundary2, pred_boundary2

def get_boundary2(row, level):
    
    """
        NOT IN USE
        Return boundaries from columns LEVEL1_GOLD, LEVEL2_GOLD, LEVEL1_PRED, LEVEL2_PRED
    """
    
    if level == "1":
        gold_column = 'LEVEL1_GOLD'
        pred_column = 'LEVEL1_PRED'
    elif level == "2":
        gold_column = 'LEVEL2_GOLD'
        pred_column = 'LEVEL2_PRED'

    gold = row[gold_column].split("-", 1)
    pred = row[pred_column].split("-", 1)

    gold_boundary = gold[0]
    pred_boundary = pred[0]

    return gold_boundary, pred_boundary

def count_golds(df, doPrint):
    
    """
        Count golds, insertions, deletions globally at both entity levels
    """
    
    if doPrint == True:
        print("Called function: count_golds...................................................")

    level1_gold = 0
    level2_gold = 0

    insertion_1 = 0
    deletion_1 = 0
    insertion_2 = 0
    deletion_2 = 0
    
    count_gold_lvl1 = defaultdict(int)
    count_gold_lvl2 = defaultdict(int)
      
    for index, row in df.iterrows():

        gold_class1, pred_class1, gold_class2, pred_class2 = get_category(row)

        count_gold_lvl1[gold_class1] += 1
        if gold_class1 != 'O':
            level1_gold += 1

        count_gold_lvl2[gold_class2] += 1
        if gold_class2 != 'O':
            level2_gold += 1

        if gold_class1 == 'O' and pred_class1 != 'O':
            insertion_1 += 1

        if gold_class1 != 'O' and pred_class1 == 'O':
            deletion_1 += 1

        if gold_class2 == 'O' and pred_class2 != 'O':
            insertion_2 += 1

        if gold_class2 != 'O' and pred_class2 == 'O':
            deletion_2 += 1
    
    if doPrint == True:            
        print("======== Total Golds ========")
        print("Level 1:")
        pretty(count_gold_lvl1, value_order=True)
        print("Total golds at level 1:", level1_gold)
        print("")
        print("Level 2:")
        pretty(count_gold_lvl2, value_order=True)
        print("Total golds at level 2:", level2_gold)
        print("")
    
    return level1_gold, level2_gold, count_gold_lvl1, count_gold_lvl2, insertion_1, deletion_1, insertion_2, deletion_2

def metrics_per_level(df, level, doPrint):
    
    """
        Computes P, R, F and specificity given a level of analysis
        Equivalent to micro-average 
    """
    
    if doPrint == True:
        print("Called function: metrics_per_level (computed on tokens)........................")

    if level == "1":
        # if doPrint == True:
        #     print("Computing at LEVEL 1 globally")
        count_dict = count_per_level(df, level)
        scores_1 = compute_scores_per_category(count_dict)
        if doPrint == True:
            # print("Scores at LEVEL 1")
            pretty(scores_1, value_order=False)
            print("")
        else:
            return scores_1

    elif level == "2":
        # if doPrint == True:
        #     print("Computing at LEVEL 2 globally")
        count_dict = count_per_level(df, level)
        scores_2 = compute_scores_per_category(count_dict)
        if doPrint == True:
            # print("Scores at LEVEL 2")
            pretty(scores_2, value_order=False)
            print("")
        else:
            return scores_2 
   
def count_per_level(df, level):

    """
        Counts TP, TN, FP, FN, given a level
    """

    count_dict = defaultdict(int)
            
    for index, row in df.iterrows():
        
        gold_class1, pred_class1, gold_class2, pred_class2 = get_category(row)

        if level == "1":
            gold_class = gold_class1
            pred_class = pred_class1
        elif level == "2":
            gold_class = gold_class2
            pred_class = pred_class2
            
        # True negative
        if gold_class == 'O' and pred_class == 'O':
            count_dict['TN'] += 1
        
        # True positive
        elif gold_class != 'O' and gold_class == pred_class:
            count_dict['TP'] += 1
        
        # False positive
        elif gold_class == 'O'and pred_class != 'O':
            count_dict['FP'] += 1
        
        # False negative
        elif gold_class != 'O' and pred_class == 'O':
            count_dict['FN'] += 1
         
        # Class errors
        elif gold_class != 'O' and pred_class != 'O' and gold_class != pred_class:
            count_dict['class_errors'] += 1

    return count_dict

def metrics_per_category(df, level, doPrint):

    """
        Computes metrics per category at given level
    """

    if doPrint == True:
        print("Called function: metrics_per_category.........................................")

    if level == "1":
        level = 'LEVEL1_GOLD'
        if doPrint == True:
            print("Computing at LEVEL 1")
        count_dict = count_per_category(df, level)
        scores_1 = compute_scores_per_category(count_dict)
        if doPrint == True:
            pretty(scores_1, value_order=False)
        else:
            return scores_1

    elif level == "2":
        if doPrint == True:
            print("Computing at LEVEL 2")
        level = 'LEVEL2_GOLD'
        count_dict = count_per_category(df, level)
        scores_2 = compute_scores_per_category(count_dict)
        if doPrint == True:
            pretty(scores_2, value_order=False)
        else:
            return scores_2

    elif level == "3": 
        if doPrint == True:
            print("Computing at LEVEL 1 and LEVEL 2")
        level = 1
        count_dict = count_per_category(df, level)
        scores_1 = compute_scores_per_category(count_dict)
        level = 2
        count_dict = count_per_category(df, level)
        scores_2 = compute_scores_per_category(count_dict)
        if doPrint == True:
            print("Scores at LEVEL1")
            pretty(scores_1, value_order=False)
            print("")
            print("Scores at LEVEL 2")
            pretty(scores_2, value_order=False) 
        else:
            return scores_1, scores_2   

def compute_scores_per_category(count_dict):

    """
        From count_per_category, computes and returns P, R, F, specificity per category
    """

    # if count_dict in not nested
    if not any(isinstance(value, dict) for value in count_dict.values()):

        scores = defaultdict(int)

        TP = count_dict['TP']
        FN = count_dict['FN']
        FP = count_dict['FP']
        TN = count_dict['TN']

        if TP == 0:

            # if there is no TP, all metrics are put to zero
            scores[i]['Precision'] = 0
            scores[i]['Recall'] = 0
            scores[i]['F-score'] = 0
            scores[i]['Specificity'] = 0

        else:

            # Recall 
            recall = TP / (TP + FN)
            "{:.2f}".format(recall)

            # Precision 
            precision = TP / (TP + FP)

            # Specificity
            specificity = TN / (TN + FP)

            # F-score
            fscore = 2 *(precision * recall) / (precision + recall)

            scores['Precision'] = precision
            scores['Recall'] = recall
            scores['F-score'] = fscore
            scores['Specificity'] = specificity

    # if count_dict is nested
    elif any(isinstance(value, dict) for value in count_dict.values()): 

        scores = defaultdict(lambda: defaultdict(int))

        for i in count_dict:

            TP = count_dict[i]['TP']
            FN = count_dict[i]['FN']
            FP = count_dict[i]['FP']
            TN = count_dict[i]['TN']

            # if there is no TP, all metrics are put to zero
            if TP == 0:

                scores[i]['Precision'] = 0
                scores[i]['Recall'] = 0
                scores[i]['F-score'] = 0
                scores[i]['Specificity'] = 0

            else:

                # Recall 
                recall = TP / (TP + FN)
                "{:.2f}".format(recall)

                # Precision 
                precision = TP / (TP + FP)

                # Specificity
                specificity = TN / (TN + FP)    

                # F-score
                fscore = 2 *(precision * recall) / (precision + recall)

                scores[i]['Precision'] = precision
                scores[i]['Recall'] = recall
                scores[i]['F-score'] = fscore
                scores[i]['Specificity'] = specificity

    return scores

def count_per_category(df, level):

    """
        Counts TP, TN, FP, FN per category at a given level
    """
    
    # print("Called function: count_per_category..........................")

    if level == "1":
        level = 'LEVEL1_GOLD'
    elif level == "2":
        level = 'LEVEL2_GOLD'

    count_dict = defaultdict(lambda: defaultdict(int))
        
    # Extract all unique gold entities at level 1    
    gold_list = df[level].tolist()
    entity_list = []
    
    for i in gold_list:
        tag = i.split("-", 1)
        if len(tag) > 1:
            category = tag[1]
            entity_list.append(category)
        else:
            category = tag[0]
            entity_list.append(category)

    entity_list = sorted(set(entity_list))
    
    # Initiate dictionary for each entity 
    count_dict = {key:defaultdict(int) for key in entity_list}
        
    for index, row in df.iterrows():
       
        gold_class1, pred_class1, gold_class2, pred_class2 = get_category(row)

        if level == 'LEVEL1_GOLD':
            gold_class = gold_class1
            pred_class = pred_class1
        elif level == 'LEVEL2_GOLD':
            gold_class = gold_class2
            pred_class = pred_class2

        key_list = entity_list.copy()          
        
        # In case of true (positive or negative)
        if gold_class == pred_class:

            # Increment key for which it is a TP
            count_dict[gold_class]['TP'] += 1

            # Increment keys for which it is a TN    
            key_list.remove(gold_class)
            for k in key_list:
                count_dict[k]['TN'] += 1

        # In case of false (negative or positive)
        elif gold_class != pred_class:

            for k in entity_list:
                if k == gold_class:
                    count_dict[gold_class]['FN'] += 1
                elif k == pred_class:    
                    count_dict[pred_class]['FP'] += 1
                elif k != gold_class and k != pred_class:
                    count_dict[k]['TN'] += 1
                    
    # pretty(count_dict, value_order=False)

    return count_dict

def class_boundary_evaluation(df, level, doPrint):
    
    """
        Counts boundary and class errors at both levels
    """
    
    if doPrint == True:
        print("Called function: class_boundary_evaluation.....................................")

    # level1_err = 0
    # level2_err = 0
    
    # boundary_err_1 = 0
    # boundary_err_2 = 0
    
    # class_err_1 = 0
    # class_err_2 = 0
    
    # boundary_class_err_1 = 0
    # boundary_class_err_2 = 0

    # class_err_dict_lvl1 = defaultdict(lambda: defaultdict(int))
    # class_err_dict_lvl2 = defaultdict(lambda: defaultdict(int))
    # boundary_err_dict = {} 

    def compute_boundaries_categories(df, level, doPrint):

        boundary_class_err = 0
        boundary_err = 0
        class_err = 0

        for index, row in df.iterrows():

            ## Les éléments comparés doivent être différents de 'O'

            gold_class, pred_class = get_category2(row, level)
            gold_boundary, pred_boundary = get_boundary2(row, level)

            # Boundary and category difference:
            if gold_boundary != pred_boundary and gold_class != pred_class:
                boundary_class_err +=1

            # Boundary difference only: 
            elif gold_boundary != pred_boundary:
                boundary_err += 1

            # Category difference only: 
            elif gold_class != pred_class:
                class_err += 1

        # if level == "1":
        #     print("Level 1")
        # elif level == "2":
        #     print("Level 2")

        if doPrint == True:
            print("\tBoundary and class errors:", boundary_class_err)
            print("\tBoundary errors:", boundary_err)
            print("\tClass errors:", class_err)
            print("")   

        return boundary_class_err, boundary_err, class_err      

    if level == "1" or level == "2":
        boundary_class_err, boundary_err, class_err = compute_boundaries_categories(df, level, doPrint)
        return boundary_class_err, boundary_err, class_err
    elif level == "3":
        level = "1"
        boundary_class_err_1, boundary_err_1, class_err_1 = compute_boundaries_categories(df, level, doPrint)
        level = "2"
        boundary_class_err_2, boundary_err_2, class_err_2 = compute_boundaries_categories(df, level, doPrint)
        return boundary_class_err_1, boundary_class_err_2, boundary_err_1, boundary_err_2, class_err_1, class_err_2

        # gold_class1, pred_class1, gold_class2, pred_class2 = get_category(row)
        # gold_boundary1, pred_boundary1, gold_boundary2, pred_boundary2 = get_boundary(row)
                    
        ## Computing boundary and class errors at Level 1 ======================================================================================
        
    #     # Boundary and category difference:
    #     if gold_boundary1 != pred_boundary1 and gold_class1 != pred_class1:
    #         boundary_class_err_1 +=1
    #         level1_err += 1
    #         class_err_dict_lvl1[gold_class1][pred_class1] += 1
        
    #     # Boundary difference only: 
    #     elif gold_boundary1 != pred_boundary1:
    #         boundary_err_1 += 1
    #         level1_err += 1
        
    #     # Category difference only: 
    #     elif gold_class1 != pred_class1:
    #         class_err_1 += 1
    #         level1_err += 1
    #         class_err_dict_lvl1[gold_class1][pred_class1] += 1
                
    #     ## Counting boundary and class errors at Level 2 ======================================================================================
                    
    #     if gold_boundary2 != pred_boundary2 and gold_class2 != pred_class2:
    #         boundary_class_err_2 +=1
    #         level2_err +=1
    #         class_err_dict_lvl2[gold_class2][pred_class2] += 1

    #     elif gold_boundary2 != pred_boundary2:
    #         boundary_err_2 +=1
    #         level2_err +=1
        
    #     elif gold_class2 != pred_class2:
    #         class_err_2 += 1
    #         level2_err +=1                
    #         class_err_dict_lvl2[gold_class2][pred_class2] += 1
                    
    # print("=================== Error Types ===================")
    # print("\t\t\tLevel 1\t\tLevel 2\t\tTotal")
    # print("Boundary\t\t", boundary_err_1, "\t\t", boundary_err_2, "\t\t", boundary_err_1+boundary_err_2)
    # print("Class\t\t\t", class_err_1, "\t\t", class_err_2, "\t\t", class_err_1+class_err_2)
    # print("Boundary and Class\t", boundary_class_err_1, "\t\t", boundary_class_err_2, "\t\t", boundary_class_err_1+boundary_class_err_2)
    # print("Total\t\t\t",level1_err, "\t\t", level2_err, "\t\t", level1_err+level2_err)
    # print("")
    # print("=================== Class Errors ===================")
    # print("Level 1:")
    # pretty(class_err_dict_lvl1, value_order=True)
    # print("")
    # print("Level 2:")
    # pretty(class_err_dict_lvl2, value_order=True)
    # print("")

    # return class_err_dict_lvl1, class_err_dict_lvl2, boundary_err_1, class_err_1, boundary_class_err_1, level1_err, boundary_err_2, class_err_2, boundary_class_err_2, level2_err
                  
def error_propagation(df):
    
    """
        Count errors between levels such as :
    
            err1_corr2 = LEVEL1 error --> LEVEL2 correct
    
            corr1_err2 = LEVEL1 correct --> LEVEL2 error
    
            err1_err2 = LEVEL1 error --> LEVEL2 error
            
            corr1_corr2_positive = LEVEL1 correct positive --> LEVEL2 correct positive
            
            corr1_corr2_negative = LEVEL1 correct negative --> LEVEL2 correct negative
            
    """
     
    print("Called function: error propagation.............................................")
        
    err1_corr2 = 0
    corr1_err2 = 0
    err1_err2 = 0
    corr1_corr2_positive = 0
    corr1_corr2_negative = 0
    
    for index, row in df.iterrows():
        
        lvl1_err = None
        lvl2_err = None
                
        # gold_1 = row['LEVEL1_GOLD'].split("-", 1)
        # pred_1 = row['LEVEL1_PRED'].split("-", 1)


        # gold_2 = row['LEVEL2_GOLD'].split("-", 1)
        # pred_2 = row['LEVEL2_PRED'].split("-", 1)

        gold_class1, pred_class1, gold_class2, pred_class2 = get_category(row)
                    
        # gold_class1 = gold_class2 = pred_class1 = pred_class2 = ""
        
        # checking error at level 1:
            
        if gold_class1 != pred_class1:
            lvl1_err = True
            
        else:
            lvl1_err = False
        
        # checking error at level 2:

            if gold_class2 != pred_class2:
                lvl2_err = True
                
            else:
                lvl2_err = False
        
        # error at level 1, no error at level 2:
        if lvl1_err == True and lvl2_err == False:
            err1_corr2 +=1
        
        # no error at level1, error at level 2:
        elif lvl1_err == False and lvl2_err == True:
            corr1_err2 += 1
        
        # error on both levels:
        elif lvl1_err == True and lvl2_err == True:
            err1_err2 += 1
        
        # no error and both levels are correctly detected as positives:
        elif lvl1_err == False and lvl2_err == False:
            corr1_corr2_positive +=1
        
        # no error and both levels are correctly detected as negatives :
        # elif lvl1_err == None and lvl2_err == None:
        #     corr1_corr2_negative += 1

    print("Error at level1, correct at level2:", err1_corr2)
    print("Correct at level1, error at level2:", corr1_err2)
    print("Errors on both levels:", err1_err2)
    print("Correct on both levels", corr1_corr2_positive)
    print("")
    
    return err1_corr2, corr1_err2, err1_err2, corr1_corr2_positive, corr1_corr2_negative

def pretty(d, value_order):
    
    """
        Print dict and nested dict
    """

    if value_order: #Si on veut l'ordre décroissant des valeurs, decreasing_dict = True
        x = 1
    else:
        x = 0

    ## If simple dict

    if not any(isinstance(value, dict) for value in d.values()):
        for key, value in sorted(d.items(), key=lambda item: item[x], reverse=value_order):


            ## Formatting floats (i.e. metrics) to display 3 decimals
            if type(value) == float:
                value = "{:.2f}".format(value)

            if key == 'O':
                next    
            elif len(key) <= 5:
                print("\t", key, "\t\t\t\t\t", value)
            elif len(key) > 15:
                print("\t", key, "\t\t\t", value)
            else:
                print("\t", key, "\t\t\t\t", value)


    ## If nested dict

    elif any(isinstance(value, dict) for value in d.values()):
        for key, value in d.items():
            if key == 'O':
                next          
            else:
                print("\t", key)
                for value,nested_value in sorted(value.items(), key=lambda item: item[x], reverse=value_order): 

                    ## Formatting floats (i.e. metrics) to display 3 decimals
                    if type(nested_value) == float:
                        nested_value = "{:.2f}".format(nested_value)

                    if len(value) <= 5:
                        print("\t\t", value, "\t\t\t\t", nested_value)
                    elif len(value) > 15:
                        print("\t\t", value, "\t\t", nested_value)
                    else:
                        print("\t\t", value, "\t\t\t", nested_value)

def slot_error_rate(boundary_class_err, boundary_err, class_err, level, doPrint):

    """
        Computes slot error rate at given level
    """

    if doPrint == True:
        print("Called function: slot_error_rate...............................................")

    level1_gold, level2_gold, count_gold_lvl1, count_gold_lvl2, insertion_1, deletion_1, insertion_2, deletion_2 = count_golds(df, doPrint = False)

    # if level == "1" or level == "2":
    #     boundary_class_err, boundary_err, class_err = class_boundary_evaluation(df, level)
    # elif level == "3":
    #     boundary_class_err_1, boundary_class_err_2, boundary_err_1, boundary_err_2, class_err_1, class_err_2 = class_boundary_evaluation(df, level)

    def compute_SER(d, i, tf, t, f, r):

        SER = (d + i + tf + 0.5 * (t + f)) / r

        return SER


    if level == "1":
        d = deletion_1
        i = insertion_1
        tf = boundary_class_err
        t = class_err
        f = boundary_err
        r = level1_gold
        SER_1 = compute_SER(d, i, tf, t, f, r)
        print("\tSER Level 1:", "{:.2f}".format(SER_1))
        print("")
        return SER_1

    elif level == "2":
        d = deletion_2
        i = insertion_2
        tf = boundary_class_err
        t = class_err
        f = boundary_err
        r = level2_gold
        SER_2 = compute_SER(d, i, tf, t, f, r)
        print("\tSER Level 2:", "{:.2f}".format(SER_2))
        print("")
        return SER_2

    # print("\tSER:", "{:.2f}".format(SER))

def recall_at_document_level(ind_df, level):

    """
        Computes recall values per entity level at document level (i.e.: how many documents are perfectly de-identified out of all documents)
    """

    print("Called function: recall_at_document_level......................................")

    count_files = 0
    total_recall = 0
    total_recall_1 = 0
    total_recall_2 = 0

    if level == "1" or level == "2":
       
        for df in ind_df:    
            
            metrics = metrics_per_level(df, level, doPrint = False)
            recall = metrics['Recall']

            if recall == 1.0:
                total_recall += 1

            count_files += 1

        doc_level_recall = total_recall/count_files

        print("\tRecall at document level for level", level, "entities (computed on tokens):", "{:.2f}".format(doc_level_recall))
        
    elif level == "3":
        for df in ind_df:    

            metrics_1 = metrics_per_level(df, level="1", doPrint = False)
            recall_1 = metrics_1['Recall']

            if recall_1 == 1.0:
                total_recall_1 += 1

            metrics_2 = metrics_per_level(df, level="2", doPrint = False)
            recall_2 = metrics_2['Recall']

            if recall_2 == 1.0:
                total_recall_2 += 1
        
            count_files += 1

        doc_level_recall_1 = total_recall_1/count_files
        doc_level_recall_2 = total_recall_2/count_files
        print("\tRecall at document level for level 1 entities (computed on tokens):", "{:.2f}".format(doc_level_recall_1))
        print("\tRecall at document level for level 2 entities (computed on tokens):", "{:.2f}".format(doc_level_recall_2))

    print("")

def macro_values(metrics_category):

    """
        Get each P and R per category from metrics_per_category, then computes the mean of all P and R
    """

    print("Called function: macro_values (computed on tokens).............................")

    count = 0
    sum_recall = 0
    sum_precision = 0
    # sum_specificity = 0

    for key, value in metrics_category.items():
        sum_recall = sum_recall + value['Recall']
        sum_precision = sum_precision + value['Precision']
        # sum_specificity = sum_specificity + value['Specificity']
        count += 1

    macro_recall = sum_recall / count
    macro_precision = sum_precision / count
    # macro_specificity = sum_specificity / count

    macro_fscore = 2 *(macro_precision * macro_recall) / (macro_precision + macro_recall)

    print("\t Macro recall", "\t\t\t\t", "{:.2f}".format(macro_recall))
    print("\t Macro precision", "\t\t\t", "{:.2f}".format(macro_precision))
    print("\t Macro F-score", "\t\t\t\t", "{:.2f}".format(macro_fscore))
    # print("Macro specificity:", "{:.2f}".format(macro_specificity))
    print("")

    return

def evaluation_per_entity(df, level):

    """
        Computes metrics for ENTITIES, at a given level
    """

    print("Called function: evaluation_per_entity.........................................")

    if level == "1":
        gold = df["LEVEL1_GOLD"].tolist()
        pred = df["LEVEL1_PRED"].tolist()

        print("")
        print(sequence_labelling.classification_report(gold, pred))

    elif level == "2":
        gold = df["LEVEL2_GOLD"].tolist()
        pred = df["LEVEL2_PRED"].tolist()

        print("")
        print(sequence_labelling.classification_report(gold, pred))

    elif level == "3":
        gold = df["LEVEL1_GOLD"].tolist()
        pred = df["LEVEL1_PRED"].tolist()

        print("================== Sequence labeling at level 1 ================")
        print("")
        print(sequence_labelling.classification_report(gold, pred))

        gold = df["LEVEL2_GOLD"].tolist()
        pred = df["LEVEL2_PRED"].tolist()

        print("==================Sequence labeling at level 2 ================")
        print("")
        print(sequence_labelling.classification_report(gold, pred))

    return

def TODO_check_boundaries_logic():
    
    """
        Checks continuity of boundary detection
    """
    
    return

def TODO_distribution_of_errors_across_docs(ind_df, level):

    """
        Checks boundary continuity between tokens/inside entities
        À CLARIFIER : faut-il considérer le rappel par document ou... autre chose ?
    """

    for df in ind_df:
        print(count_per_level(df, level))

    return

def TODO_mean_average_precision():

    return

def TODO_r_precision():

    return

# directory = "/home/gianola/Documents/MAPA/outputs/generated_predictions_for_eval_20210727/generated_predictions_for_eval_20210727/COUR_CASSATION1::LEGAL::fr::multi::True::42::mapa_v1::IOB"

# Int parameter determines what is calculated: 1 for Level 1 only, 2 for Level 2 only, and 3 for both levels

parser = argparse.ArgumentParser(
        description="Given a directory of Vicomtech output files, count the error types")
parser.add_argument("dir_path", type=str,
                        help="Path to the directory containing the output .tsv files from Vicomtech")
parser.add_argument("level", type=str,
                        help="Level of analysis: 1 for LEVEL1, 2 for LEVEL2, 3 for both levels")
args = parser.parse_args()
directory = args.dir_path
level = args.level

print("")
if level == "1" or level == "2":
    print("Analysis performed at level", level)
    print("")
elif level == "3":
    print("Analysis performed at both levels")
    print("")

df = load_TSV_to_DF(directory)

ind_df = load_each_file(directory)

recall_at_document_level(ind_df, level)

# # distribution_of_errors_across_docs(ind_df, level)

if level == "1" or level == "2":
    boundary_class_err, boundary_err, class_err = class_boundary_evaluation(df, level, doPrint=False)
    slot_error_rate(boundary_class_err, boundary_err, class_err, level, doPrint = True)
    metrics_per_level(df, level, doPrint = True)
    metrics_category = metrics_per_category(df, level, doPrint = False)
    macro_values(metrics_category)

elif level == "3":
    boundary_class_err_1, boundary_class_err_2, boundary_err_1, boundary_err_2, class_err_1, class_err_2 = class_boundary_evaluation(df, level, doPrint=False)
    SER_1 = slot_error_rate(boundary_class_err_1, boundary_err_1, class_err_1, "1", doPrint = True)
    SER_2 = slot_error_rate(boundary_class_err_2, boundary_err_2, class_err_2, "2", doPrint = False)

    metrics_per_level(df, level, doPrint = True)
    metrics_1, metrics_2 = metrics_per_category(df, level, doPrint = False)
    macro_values(metrics_1)
    macro_values(metrics_2)
    
evaluation_per_entity(df, level)
