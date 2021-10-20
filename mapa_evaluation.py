import csv
import os
import fnmatch
import pandas as pd
import argparse
from collections import defaultdict, OrderedDict
import glob

def loadTSVtoDF(directory):

    ##
    ## Load all files of given dict to a single dataframe
    ##

    # print("File loading: loadTSVtoDF......................................................")

    files = glob.glob(directory + "/*.tsv")
    dfs = [pd.read_csv(f, sep='\t', skiprows=1, quoting=csv.QUOTE_NONE, usecols=['TOKEN','LEVEL1_GOLD','LEVEL2_GOLD','LEVEL1_PRED','LEVEL2_PRED','TOKEN_SPAN']) for f in files]

    df = pd.concat(dfs,ignore_index=True)
import csv
import os
import fnmatch
import pandas as pd
import argparse
from collections import defaultdict, OrderedDict
import glob 

def load_TSV_to_DF(directory):

    ##
    ## Load all files of given dict to a single dataframe
    ##

    # print("File loading: load_TSV_to_DF......................................................")

    files = glob.glob(directory + "/*.tsv")
    dfs = [pd.read_csv(f, sep='\t', skiprows=1, quoting=csv.QUOTE_NONE, usecols=['TOKEN','LEVEL1_GOLD','LEVEL2_GOLD','LEVEL1_PRED','LEVEL2_PRED','TOKEN_SPAN']) for f in files]

    df = pd.concat(dfs,ignore_index=True)

    return df

def get_category(row):
    
    ##
    ## Return categories from columns LEVEL1_GOLD, LEVEL2_GOLD, LEVEL1_PRED, LEVEL2_PRED
    ## If row is 'O', returns 'O'
    ##
    
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
    
    ##
    ## Return categories from columns LEVEL1_GOLD, LEVEL2_GOLD, LEVEL1_PRED, LEVEL2_PRED
    ## If row is 'O', returns 'O'
    ##
    
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
    
    ##
    ## Return boundaries from columns LEVEL1_GOLD, LEVEL2_GOLD, LEVEL1_PRED, LEVEL2_PRED
    ##
    
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
    
    ##
    ## Return boundaries from columns LEVEL1_GOLD, LEVEL2_GOLD, LEVEL1_PRED, LEVEL2_PRED
    ##
    
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
    
    ##
    ## Count golds globally and per category at both entity levels
    ##
    
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

def metrics_per_level(df, level):
    
    ##
    ## Computes P, R, F and specificity for each level 
    ##
    
    print("Called function: metrics_per_level.............................................")

    if level == "1":
        level = 'LEVEL1_GOLD'
        print("Computing at LEVEL 1 globally")
        count_dict = count_per_level(df, level)
        scores_1 = compute_scores(count_dict)
        print("Scores at LEVEL 1")
        pretty(scores_1, value_order=False)
        print("")

    elif level == "2":
        print("Computing at LEVEL 2 globally")
        level = 'LEVEL2_GOLD'
        count_dict = count_per_level(df, level)
        scores_2 = compute_scores(count_dict)
        print("Scores at LEVEL 2")
        pretty(scores_2, value_order=False)
        print("")

    elif level == "3": 
        print("Computing at LEVEL 1 and LEVEL 2 globally")
        level = 1
        count_dict = count_per_level(df, level)
        scores_1 = compute_scores(count_dict)
        level = 2
        count_dict = count_per_level(df, level)
        scores_2 = compute_scores(count_dict)
        print("Scores at LEVEL 1")
        pretty(scores_1, value_order=False)
        print("")
        print("Scores at LEVEL 2")
        pretty(scores_2, value_order=False)
        print("")  
   
    return

def count_per_level(df, level):

    if level == 1:
        level = 'LEVEL1_GOLD'
    elif level == 2:
        level = 'LEVEL2_GOLD'

    count_dict = defaultdict(int)
            
    for index, row in df.iterrows():
        
        gold_class1, pred_class1, gold_class2, pred_class2 = get_category(row)

        if level == 'LEVEL1_GOLD':
            gold_class = gold_class1
            pred_class = pred_class1
        elif level == 'LEVEL2_GOLD':
            gold_class = gold_class2
            pred_class = pred_class2
            
        ## Counting for level 1
        
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

def metrics_per_category(df, level):

    ##
    ##  Computes metrics per category
    ##

    print("Called function: metrics_per_category..........................................")

    if level == "1":
        level = 'LEVEL1_GOLD'
        print("Computing at LEVEL 1")
        count_dict = count_per_category(df, level)
        scores_1 = compute_scores(count_dict)
        pretty(scores_1, value_order=False)

    elif level == "2":
        print("Computing at LEVEL 2")
        level = 'LEVEL2_GOLD'
        count_dict = count_per_category(df, level)
        scores_2 = compute_scores(count_dict)
        pretty(scores_2, value_order=False)

    elif level == "3": 
        print("Computing at LEVEL 1 and LEVEL 2")
        level = 1
        count_dict = count_per_category(df, level)
        scores_1 = compute_scores(count_dict)
        level = 2
        count_dict = count_per_category(df, level)
        scores_2 = compute_scores(count_dict)
        print("Scores at LEVEL1")
        pretty(scores_1, value_order=False)
        print("")
        print("Scores at LEVEL 2")
        pretty(scores_2, value_order=False)    

    return

def compute_scores(count_dict):

    if not any(isinstance(value, dict) for value in count_dict.values()):

        scores = defaultdict(int)

        TP = count_dict['TP']
        FN = count_dict['FN']
        FP = count_dict['FP']
        TN = count_dict['TN']

        if TP == 0:

            scores[i]['Precision'] = 0
            scores[i]['Recall'] = 0
            scores[i]['F-score'] = 0
            specificity[i]['Specificity'] = 0

        else:

            # Recall 
            recall = TP / (TP + FN)
            "{:.3f}".format(recall)

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

    elif any(isinstance(value, dict) for value in count_dict.values()): 

        scores = defaultdict(lambda: defaultdict(int))

        for i in count_dict:

            TP = count_dict[i]['TP']
            FN = count_dict[i]['FN']
            FP = count_dict[i]['FP']
            TN = count_dict[i]['TN']

            if TP == 0:

                scores[i]['Precision'] = 0
                scores[i]['Recall'] = 0
                scores[i]['F-score'] = 0
                scores[i]['Specificity'] = 0

            else:

                # Recall 
                recall = TP / (TP + FN)
                "{:.3f}".format(recall)

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

    ##
    ## Computes metrics for all files of a directory per category
    ##
    
    # print("Called function: count_per_category..........................")

    if level == 1:
        level = 'LEVEL1_GOLD'
    elif level == 2:
        level = 'LEVEL2_GOLD'

    count_dict = defaultdict(lambda: defaultdict(int))
        
    ## Extract all unique gold entities at level 1    
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
    
    ## Initiate dictionary for each entity 
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

def class_boundary_evaluation(df, level):
    
    ##
    ## Counts boundary and class errors at both levels
    ##
    
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

    def compute_boundaries_categories(df, level):

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

        if level == "1":
            print("Level 1")
        elif level == "2":
            print("Level 2")

        print("\tBoundary and class errors:", boundary_class_err)
        print("\tBoundary errors:", boundary_err)
        print("\tClass errors:", class_err)
        print("")   

        return boundary_class_err, boundary_err, class_err      

    if level == "1" or level == "2":
        boundary_class_err, boundary_err, class_err = compute_boundaries_categories(df, level)
        return boundary_class_err, boundary_err, class_err
    elif level == "3":
        level = "1"
        boundary_class_err_1, boundary_err_1, class_err_1 = compute_boundaries_categories(df, level)
        level = "2"
        boundary_class_err_2, boundary_err_2, class_err_2 = compute_boundaries_categories(df, level)
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
    
    ##
    ## Count errors between levels such as :
    ##
    ## err1_corr2 = LEVEL1 error --> LEVEL2 correct
    ## 
    ## corr1_err2 = LEVEL1 correct --> LEVEL2 error
    ##
    ## err1_err2 = LEVEL1 error --> LEVEL2 error
    ##
    ## corr1_corr2_positive = LEVEL1 correct positive --> LEVEL2 correct positive
    ##
    ## corr1_corr2_negative = LEVEL1 correct negative --> LEVEL2 correct negative
    ##
    ##
     
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
    
    ##
    ## Print dict and nested dict 
    ##

    if value_order: #Si on veut l'ordre décroissant des valeurs, decreasing_dict = True
        x = 1
    else:
        x = 0

    ## If simple dict

    if not any(isinstance(value, dict) for value in d.values()):
        for key, value in sorted(d.items(), key=lambda item: item[x], reverse=value_order):


            ## Formatting floats (i.e. metrics) to display 3 decimals
            if type(value) == float:
                value = "{:.3f}".format(value)

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
                        nested_value = "{:.3f}".format(nested_value)

                    if len(value) <= 5:
                        print("\t\t", value, "\t\t\t\t", nested_value)
                    elif len(value) > 15:
                        print("\t\t", value, "\t\t", nested_value)
                    else:
                        print("\t\t", value, "\t\t\t", nested_value)

def slot_error_rate(boundary_class_err, boundary_err, class_err, level):

    print("Called function: slot_error_rate, level:", level, "....................................")

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
        print("SER Level 1:", "{:.3f}".format(SER_1))
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
        print("SER Level 2:", "{:.3f}".format(SER_2))
        print("")
        return SER_2

    print("SER:", "{:.3f}".format(SER))

def TODO_check_boundaries_logic():
    
    ##
    ##  Checks continuity of boundary detection
    ##
    
    return

def TODO_metrics_at_document_level():

    return

# directory = "/home/gianola/Documents/MAPA/outputs/generated_predictions_for_eval_20210727/generated_predictions_for_eval_20210727/COUR_CASSATION1::LEGAL::fr::multi::True::42::mapa_v1::IOB"

parser = argparse.ArgumentParser(
        description="Given a directory of Vicomtech output files, count the error types")
parser.add_argument("dir_path", type=str,
                        help="Path to the directory containing the output .tsv files from Vicomtech")
parser.add_argument("level", type=str,
                        help="Level of analysis: 1 for LEVEL1, 2 for LEVEL2, 3 for both levels")
args = parser.parse_args()
directory = args.dir_path
level = args.level

df = load_TSV_to_DF(directory)
# level = "3"

# count_golds(df, doPrint=True)

if level == "1" or level == "2":
    boundary_class_err, boundary_err, class_err = class_boundary_evaluation(df, level)
    slot_error_rate(boundary_class_err, boundary_err, class_err, level)

elif level == "3":
    boundary_class_err_1, boundary_class_err_2, boundary_err_1, boundary_err_2, class_err_1, class_err_2 = class_boundary_evaluation(df, level)
    SER_1 = slot_error_rate(boundary_class_err_1, boundary_err_1, class_err_1, "1")
    SER_2 = slot_error_rate(boundary_class_err_2, boundary_err_2, class_err_2, "2")

## For metrics functions, int parameter determines what is calculated: 1 for Level 1 only, 2 for Level 2 only, and 3 for both levels
metrics_per_level(df, level)
metrics_per_category(df, level)

    return df

def get_category(row):
    
    ##
    ## Return categories from columns LEVEL1_GOLD, LEVEL2_GOLD, LEVEL1_PRED, LEVEL2_PRED
    ## If row is 'O', returns 'O'
    ##
    
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

    return gold_class1, pred_class1, gold_class2, pred_class2

def get_boundary(row):
    
    ##
    ## Return boundaries from columns LEVEL1_GOLD, LEVEL2_GOLD, LEVEL1_PRED, LEVEL2_PRED
    ##
    
    gold_1 = row['LEVEL1_GOLD'].split("-", 1)
    pred_1 = row['LEVEL1_PRED'].split("-", 1)

    gold_2 = row['LEVEL2_GOLD'].split("-", 1)
    pred_2 = row['LEVEL2_PRED'].split("-", 1)
    
    gold_boundary1 = gold_1[0]
    pred_boundary1 = pred_1[0]
    gold_boundary2 = gold_2[0]
    pred_boundary2 = pred_2[0]

    return gold_boundary1, pred_boundary1, gold_boundary2, pred_boundary2

def count_golds(df):
    
    ##
    ## Count golds globally and per category at both entity levels
    ##
    
    print("Called function: count_golds...................................................")

    level1_gold = 0
    level2_gold = 0
    
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
                
    print("======== Total Golds ========")
    print("Level 1:")
    pretty(count_gold_lvl1, value_order=True)
    print("Total golds at level 1:", level1_gold)
    print("")
    print("Level 2:")
    pretty(count_gold_lvl2, value_order=True)
    print("Total golds at level 2:", level2_gold)
    print("")
    
    return level1_gold, level2_gold, count_gold_lvl1, count_gold_lvl2

def metrics_per_level(df, level):
    
    ##
    ## Computes P, R, F for each category at level 1 and level 2
    ##
    
    print("Called function: metrics_per_level.............................................")

    if level == "1":
        level = 'LEVEL1_GOLD'
        print("Computing at LEVEL 1 globally")
        count_dict = count_per_level(df, level)
        scores_1 = compute_scores(count_dict)
        print("Scores at LEVEL 1")
        pretty(scores_1, value_order=False)
        print("")

    elif level == "2":
        print("Computing at LEVEL 2 globally")
        level = 'LEVEL2_GOLD'
        count_dict = count_per_level(df, level)
        scores_2 = compute_scores(count_dict)
        print("Scores at LEVEL 2")
        pretty(scores_2, value_order=False)
        print("")

    elif level == "3": 
        print("Computing at LEVEL 1 and LEVEL 2 globally")
        level = 1
        count_dict = count_per_level(df, level)
        scores_1 = compute_scores(count_dict)
        level = 2
        count_dict = count_per_level(df, level)
        scores_2 = compute_scores(count_dict)
        print("Scores at LEVEL 1")
        pretty(scores_1, value_order=False)
        print("")
        print("Scores at LEVEL 2")
        pretty(scores_2, value_order=False)
        print("")  
   
    return

def count_per_level(df, level):

    if level == 1:
        level = 'LEVEL1_GOLD'
    elif level == 2:
        level = 'LEVEL2_GOLD'

    count_dict = defaultdict(int)
            
    for index, row in df.iterrows():
        
        gold_class1, pred_class1, gold_class2, pred_class2 = get_category(row)

        if level == 'LEVEL1_GOLD':
            gold_class = gold_class1
            pred_class = pred_class1
        elif level == 'LEVEL2_GOLD':
            gold_class = gold_class2
            pred_class = pred_class2
            
        ## Counting for level 1
        
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

def metrics_per_category(df, level):

    ##
    ##  Computes metrics per category
    ##

    print("Called function: metrics_per_category..........................................")

    if level == "1":
        level = 'LEVEL1_GOLD'
        print("Computing at LEVEL 1")
        count_dict = count_per_category(df, level)
        scores_1 = compute_scores(count_dict)
        pretty(scores_1, value_order=False)

    elif level == "2":
        print("Computing at LEVEL 2")
        level = 'LEVEL2_GOLD'
        count_dict = count_per_category(df, level)
        scores_2 = compute_scores(count_dict)
        pretty(scores_2, value_order=False)

    elif level == "3": 
        print("Computing at LEVEL 1 and LEVEL 2")
        level = 1
        count_dict = count_per_category(df, level)
        scores_1 = compute_scores(count_dict)
        level = 2
        count_dict = count_per_category(df, level)
        scores_2 = compute_scores(count_dict)
        print("Scores at LEVEL1")
        pretty(scores_1, value_order=False)
        print("")
        print("Scores at LEVEL 2")
        pretty(scores_2, value_order=False)    

    return

def compute_scores(count_dict):

    if not any(isinstance(value, dict) for value in count_dict.values()):

        scores = defaultdict(int)

        TP = count_dict['TP']
        FN = count_dict['FN']
        FP = count_dict['FP']
        TN = count_dict['TN']

        if TP == 0:

            scores[i]['Precision'] = 0
            scores[i]['Recall'] = 0
            scores[i]['F-score'] = 0
            specificity[i]['Specificity'] = 0

        else:

            # Recall 
            recall = TP / (TP + FN)
            "{:.3f}".format(recall)

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

    elif any(isinstance(value, dict) for value in count_dict.values()): 

        scores = defaultdict(lambda: defaultdict(int))

        for i in count_dict:

            TP = count_dict[i]['TP']
            FN = count_dict[i]['FN']
            FP = count_dict[i]['FP']
            TN = count_dict[i]['TN']

            if TP == 0:

                scores[i]['Precision'] = 0
                scores[i]['Recall'] = 0
                scores[i]['F-score'] = 0
                scores[i]['Specificity'] = 0

            else:

                # Recall 
                recall = TP / (TP + FN)
                "{:.3f}".format(recall)

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

    ##
    ## Computes metrics for all files of a directory per category
    ##
    
    # print("Called function: count_per_category..........................")

    if level == 1:
        level = 'LEVEL1_GOLD'
    elif level == 2:
        level = 'LEVEL2_GOLD'

    count_dict = defaultdict(lambda: defaultdict(int))
        
    ## Extract all unique gold entities at level 1    
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
    
    ## Initiate dictionary for each entity 
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

def class_boundary_evaluation(df):
    
    ##
    ## Counts boundary and class errors at both levels
    ##
    
    print("Called function: class_boundary_evaluation.....................................")

    level1_err = 0
    level2_err = 0
    
    boundary_err_1 = 0
    boundary_err_2 = 0
    
    class_err_1 = 0
    class_err_2 = 0
    
    boundary_class_err_1 = 0
    boundary_class_err_2 = 0

    class_err_dict_lvl1 = defaultdict(lambda: defaultdict(int))
    class_err_dict_lvl2 = defaultdict(lambda: defaultdict(int))
    boundary_err_dict = {} 
    
    for index, row in df.iterrows():
                    
        ## Computing boundary and class errors at Level 1 ======================================================================================
        
        if row['LEVEL1_GOLD'] != 'O' and row['LEVEL1_PRED'] != 'O' and row['LEVEL1_GOLD'] != row['LEVEL1_PRED']:
            
            gold = row['LEVEL1_GOLD'].split("-", 1)
            pred = row['LEVEL1_PRED'].split("-", 1)
            
            # get boundary:
            gold_boundary = gold[0]
            pred_boundary = pred[0]
            
            # get category:
            gold_class = gold[1]
            pred_class = pred[1]
            
            # Boundary and category difference:
            if gold_boundary != pred_boundary and gold_class != pred_class:
                boundary_class_err_1 +=1
                level1_err += 1
                class_err_dict_lvl1[gold_class][pred_class] += 1
            
            # Boundary difference only: 
            elif gold_boundary != pred_boundary:
                boundary_err_1 += 1
                level1_err += 1
            
            # Category difference only: 
            elif gold_class != pred_class:
                class_err_1 += 1
                level1_err += 1
                class_err_dict_lvl1[gold_class][pred_class] += 1
                
        ## Counting boundary and class errors at Level 2 ======================================================================================
        
        if row['LEVEL2_GOLD'] != 'O' and row['LEVEL2_PRED'] != 'O' and row['LEVEL2_GOLD'] != row['LEVEL2_PRED']:
            gold = row['LEVEL2_GOLD'].split("-", 1)
            pred = row['LEVEL2_PRED'].split("-", 1)
            
            # Get boundary:
            gold_boundary = gold[0]
            pred_boundary = pred[0]
                            
            # Get category:
            gold_class = gold[1]
            pred_class = pred[1]
            
            if gold_boundary != pred_boundary and gold_class != pred_class:
                boundary_class_err_2 +=1
                level2_err +=1
                class_err_dict_lvl2[gold_class][pred_class] += 1

            elif gold_boundary != pred_boundary:
                boundary_err_2 +=1
                level2_err +=1
            
            elif gold_class != pred_class:
                class_err_2 += 1
                level2_err +=1                
                class_err_dict_lvl2[gold_class][pred_class] += 1
                    
    print("=================== Error Types ===================")
    print("\t\t\tLevel 1\t\tLevel 2\t\tTotal")
    print("Boundary\t\t", boundary_err_1, "\t\t", boundary_err_2, "\t\t", boundary_err_1+boundary_err_2)
    print("Class\t\t\t", class_err_1, "\t\t", class_err_2, "\t\t", class_err_1+class_err_2)
    print("Boundary and Class\t", boundary_class_err_1, "\t\t", boundary_class_err_2, "\t\t", boundary_class_err_1+boundary_class_err_2)
    print("Total\t\t\t",level1_err, "\t\t", level2_err, "\t\t", level1_err+level2_err)
    print("")
    print("=================== Class Errors ===================")
    print("Level 1:")
    pretty(class_err_dict_lvl1, value_order=True)
    print("")
    print("Level 2:")
    pretty(class_err_dict_lvl2, value_order=True)
    print("")

    return class_err_dict_lvl1, class_err_dict_lvl2, boundary_err_1, class_err_1, boundary_class_err_1, level1_err, boundary_err_2, class_err_2, boundary_class_err_2, level2_err
                  
def error_propagation(df):
    
    ##
    ## Count errors between levels such as :
    ##
    ## err1_corr2 = LEVEL1 error --> LEVEL2 correct
    ## 
    ## corr1_err2 = LEVEL1 correct --> LEVEL2 error
    ##
    ## err1_err2 = LEVEL1 error --> LEVEL2 error
    ##
    ## corr1_corr2_positive = LEVEL1 correct positive --> LEVEL2 correct positive
    ##
    ## corr1_corr2_negative = LEVEL1 correct negative --> LEVEL2 correct negative
    ##
    ##
     
    print("Called function: error propagation.............................................")
        
    err1_corr2 = 0
    corr1_err2 = 0
    err1_err2 = 0
    corr1_corr2_positive = 0
    corr1_corr2_negative = 0
    
    for index, row in df.iterrows():
        
        lvl1_err = None
        lvl2_err = None
        
        token = row['TOKEN']
        
        gold_1 = row['LEVEL1_GOLD'].split("-", 1)
        pred_1 = row['LEVEL1_PRED'].split("-", 1)


        gold_2 = row['LEVEL2_GOLD'].split("-", 1)
        pred_2 = row['LEVEL2_PRED'].split("-", 1)
                    
        gold_class1 = gold_class2 = pred_class1 = pred_class2 = ""
        
        # checking error at level 1:
        if len(gold_1) > 1 and len(pred_1) > 1:
            gold_class1 = gold_1[1]
            pred_class1 = pred_1[1]
            
            if gold_class1 != pred_class1:
                lvl1_err = True
                
            else:
                lvl1_err = False
        
        # checking error at level 2:
        if len(gold_2) > 1 and len(pred_2) > 1:
            
            gold_class2 = gold_2[1]
            pred_class2 = pred_2[1]
            
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
        elif lvl1_err == None and lvl2_err == None:
            corr1_corr2_negative += 1

    print("Error at level1, correct at level2:", err1_corr2)
    print("Correct at level1, error at level2:", corr1_err2)
    print("")
    
    return err1_corr2, corr1_err2, err1_err2, corr1_corr2_positive, corr1_corr2_negative

def pretty(d, value_order):
    
    ##
    ## Print dict and nested dict 
    ##

    if value_order: #Si on veut l'ordre décroissant des valeurs, decreasing_dict = True
        x = 1
    else:
        x = 0

    ## If simple dict

    if not any(isinstance(value, dict) for value in d.values()):
        for key, value in sorted(d.items(), key=lambda item: item[x], reverse=value_order):


            ## Formatting floats (i.e. metrics) to display 3 decimals
            if type(value) == float:
                value = "{:.3f}".format(value)

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
                        nested_value = "{:.3f}".format(nested_value)

                    if len(value) <= 5:
                        print("\t\t", value, "\t\t\t\t", nested_value)
                    elif len(value) > 15:
                        print("\t\t", value, "\t\t", nested_value)
                    else:
                        print("\t\t", value, "\t\t\t", nested_value)

def TODO_check_boundaries_logic():
    
    ##
    ##  Checks continuity of boundary detection
    ##
    
    return

def TODO_slot_error_rate():

    return

# directory = "/home/gianola/Documents/MAPA/outputs/generated_predictions_for_eval_20210727/generated_predictions_for_eval_20210727/MEDDOCAN::MEDICAL::es::multi::True::42::mapa_v1::IOB"

parser = argparse.ArgumentParser(
        description="Given a directory of Vicomtech output files, count the error types")
parser.add_argument("dir_path", type=str,
                        help="Path to the directory containing the output .tsv files from Vicomtech")
parser.add_argument("level", type=str,
                        help="Level of analysis: 1 for LEVEL1, 2 for LEVEL2, 3 for both levels")
args = parser.parse_args()
directory = args.dir_path
level = args.level

df = loadTSVtoDF(directory)

class_boundary_evaluation(df)

## For metrics functions, int parameter determines what is calculated: 1 for Level 1 only, 2 for Level 2 only, and 3 for both levels
metrics_per_level(df, level)
metrics_per_category(df, level)
