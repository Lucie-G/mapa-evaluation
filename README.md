# mapa-evaluation

# Usage:

mapa_evaluation.py path/to/directory level

# Ouput example:

Called function: class_boundary_evaluation.....................................  
=================== Error Types ===================  
 &nbsp; &nbsp; &nbsp;Level 1 &nbsp; &nbsp;Level 2 &nbsp; &nbsp;Total  
Boundary &nbsp; &nbsp; 19  &nbsp; &nbsp; 0  &nbsp; &nbsp; 19  
Class &nbsp; &nbsp; &nbsp; 15  &nbsp; &nbsp; 5  &nbsp; &nbsp; 20  
Boundary and Class &nbsp; 0  &nbsp; &nbsp; 3  &nbsp; &nbsp; 3  
Total &nbsp; &nbsp; &nbsp; 34  &nbsp; &nbsp; 8  &nbsp; &nbsp; 42  
  
=================== Class Errors ===================  
Level 1:  
 &nbsp; PERSON  
 &nbsp; &nbsp; ORGANISATION  &nbsp; &nbsp; &nbsp; 6  
 &nbsp; ORGANISATION  
 &nbsp; &nbsp; ADDRESS  &nbsp; &nbsp; &nbsp; 8  
 &nbsp; &nbsp; PERSON  &nbsp; &nbsp; &nbsp; 1  
  
Level 2:  
 &nbsp; given name - female  
 &nbsp; &nbsp; given name - male  &nbsp; &nbsp; 2  
 &nbsp; &nbsp; family name  &nbsp; &nbsp; &nbsp; 2  
 &nbsp; postcode  
 &nbsp; &nbsp; street  &nbsp; &nbsp; &nbsp; 1  
 &nbsp; standard abbreviation  
 &nbsp; &nbsp; day  &nbsp; &nbsp; &nbsp; &nbsp; 1  
 &nbsp; &nbsp; month  &nbsp; &nbsp; &nbsp; &nbsp; 1  
 &nbsp; &nbsp; year  &nbsp; &nbsp; &nbsp; &nbsp; 1  
  
Called function: metrics_per_level.............................................  
Computing at LEVEL 1 and LEVEL 2 globally  
Scores at LEVEL 1  
 &nbsp; F-score  &nbsp; &nbsp; &nbsp; &nbsp; 0.951  
 &nbsp; Precision  &nbsp; &nbsp; &nbsp; &nbsp; 0.958  
 &nbsp; Recall  &nbsp; &nbsp; &nbsp; &nbsp; 0.945  
 &nbsp; Specificity  &nbsp; &nbsp; &nbsp; &nbsp; 0.997  
  
Scores at LEVEL 2  
 &nbsp; F-score  &nbsp; &nbsp; &nbsp; &nbsp; 0.969  
 &nbsp; Precision  &nbsp; &nbsp; &nbsp; &nbsp; 0.993  
 &nbsp; Recall  &nbsp; &nbsp; &nbsp; &nbsp; 0.946  
 &nbsp; Specificity  &nbsp; &nbsp; &nbsp; &nbsp; 1.000  
  
Called function: metrics_per_category..........................................  
Computing at LEVEL 1 and LEVEL 2  
Scores at LEVEL1  
 &nbsp; ADDRESS  
 &nbsp; &nbsp; F-score  &nbsp; &nbsp; &nbsp; 0.950  
 &nbsp; &nbsp; Precision  &nbsp; &nbsp; &nbsp; 0.915  
 &nbsp; &nbsp; Recall  &nbsp; &nbsp; &nbsp; 0.989  
 &nbsp; &nbsp; Specificity  &nbsp; &nbsp; &nbsp; 0.999  
 &nbsp; AMOUNT  
 &nbsp; &nbsp; F-score  &nbsp; &nbsp; &nbsp; 0.390  
 &nbsp; &nbsp; Precision  &nbsp; &nbsp; &nbsp; 0.667  
 &nbsp; &nbsp; Recall  &nbsp; &nbsp; &nbsp; 0.276  
 &nbsp; &nbsp; Specificity  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; DATE  
 &nbsp; &nbsp; F-score  &nbsp; &nbsp; &nbsp; 0.978  
 &nbsp; &nbsp; Precision  &nbsp; &nbsp; &nbsp; 0.997  
 &nbsp; &nbsp; Recall  &nbsp; &nbsp; &nbsp; 0.959  
 &nbsp; &nbsp; Specificity  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; ORGANISATION  
 &nbsp; &nbsp; F-score  &nbsp; &nbsp; &nbsp; 0.794  
 &nbsp; &nbsp; Precision  &nbsp; &nbsp; &nbsp; 0.757  
 &nbsp; &nbsp; Recall  &nbsp; &nbsp; &nbsp; 0.835  
 &nbsp; &nbsp; Specificity  &nbsp; &nbsp; &nbsp; 0.997  
 &nbsp; PERSON  
 &nbsp; &nbsp; F-score  &nbsp; &nbsp; &nbsp; 0.991  
 &nbsp; &nbsp; Precision  &nbsp; &nbsp; &nbsp; 0.997  
 &nbsp; &nbsp; Recall  &nbsp; &nbsp; &nbsp; 0.984  
 &nbsp; &nbsp; Specificity  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; TIME  
 &nbsp; &nbsp; F-score  &nbsp; &nbsp; &nbsp; 0  
 &nbsp; &nbsp; Precision  &nbsp; &nbsp; &nbsp; 0  
 &nbsp; &nbsp; Recall  &nbsp; &nbsp; &nbsp; 0  
 &nbsp; &nbsp; Specificity  &nbsp; &nbsp; &nbsp; 0  
  
Scores at LEVEL 2  
 &nbsp; ROLE  
 &nbsp; &nbsp; F-score  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; &nbsp; Precision  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; &nbsp; Recall  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; &nbsp; Specificity  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; city  
 &nbsp; &nbsp; F-score  &nbsp; &nbsp; &nbsp; 0.969  
 &nbsp; &nbsp; Precision  &nbsp; &nbsp; &nbsp; 0.939  
 &nbsp; &nbsp; Recall  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; &nbsp; Specificity  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; day  
 &nbsp; &nbsp; F-score  &nbsp; &nbsp; &nbsp; 0.995  
 &nbsp; &nbsp; Precision  &nbsp; &nbsp; &nbsp; 0.991  
 &nbsp; &nbsp; Recall  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; &nbsp; Specificity  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; family name  
 &nbsp; &nbsp; F-score  &nbsp; &nbsp; &nbsp; 0.974  
 &nbsp; &nbsp; Precision  &nbsp; &nbsp; &nbsp; 0.983  
 &nbsp; &nbsp; Recall  &nbsp; &nbsp; &nbsp; 0.966  
 &nbsp; &nbsp; Specificity  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; given name - female  
 &nbsp; &nbsp; F-score  &nbsp; &nbsp; &nbsp; 0  
 &nbsp; &nbsp; Precision  &nbsp; &nbsp; &nbsp; 0  
 &nbsp; &nbsp; Recall  &nbsp; &nbsp; &nbsp; 0  
 &nbsp; &nbsp; Specificity  &nbsp; &nbsp; &nbsp; 0  
 &nbsp; given name - male  
 &nbsp; &nbsp; F-score  &nbsp; &nbsp; &nbsp; 0.917  
 &nbsp; &nbsp; Precision  &nbsp; &nbsp; &nbsp; 0.846  
 &nbsp; &nbsp; Recall  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; &nbsp; Specificity  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; month  
 &nbsp; &nbsp; F-score  &nbsp; &nbsp; &nbsp; 0.995  
 &nbsp; &nbsp; Precision  &nbsp; &nbsp; &nbsp; 0.990  
 &nbsp; &nbsp; Recall  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; &nbsp; Specificity  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; postcode  
 &nbsp; &nbsp; F-score  &nbsp; &nbsp; &nbsp; 0.952  
 &nbsp; &nbsp; Precision  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; &nbsp; Recall  &nbsp; &nbsp; &nbsp; 0.909  
 &nbsp; &nbsp; Specificity  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; standard abbreviation  
 &nbsp; &nbsp; F-score  &nbsp; &nbsp; &nbsp; 0.812  
 &nbsp; &nbsp; Precision  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; &nbsp; Recall  &nbsp; &nbsp; &nbsp; 0.683  
 &nbsp; &nbsp; Specificity  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; street  
 &nbsp; &nbsp; F-score  &nbsp; &nbsp; &nbsp; 0.978  
 &nbsp; &nbsp; Precision  &nbsp; &nbsp; &nbsp; 0.978  
 &nbsp; &nbsp; Recall  &nbsp; &nbsp; &nbsp; 0.978  
 &nbsp; &nbsp; Specificity  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; title  
 &nbsp; &nbsp; F-score  &nbsp; &nbsp; &nbsp; 0.996  
 &nbsp; &nbsp; Precision  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; &nbsp; Recall  &nbsp; &nbsp; &nbsp; 0.993  
 &nbsp; &nbsp; Specificity  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; unit  
 &nbsp; &nbsp; F-score  &nbsp; &nbsp; &nbsp; 0.333  
 &nbsp; &nbsp; Precision  &nbsp; &nbsp; &nbsp; 0.750  
 &nbsp; &nbsp; Recall  &nbsp; &nbsp; &nbsp; 0.214  
 &nbsp; &nbsp; Specificity  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; value  
 &nbsp; &nbsp; F-score  &nbsp; &nbsp; &nbsp; 0.400  
 &nbsp; &nbsp; Precision  &nbsp; &nbsp; &nbsp; 0.800  
 &nbsp; &nbsp; Recall  &nbsp; &nbsp; &nbsp; 0.267  
 &nbsp; &nbsp; Specificity  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; year  
 &nbsp; &nbsp; F-score  &nbsp; &nbsp; &nbsp; 0.992  
 &nbsp; &nbsp; Precision  &nbsp; &nbsp; &nbsp; 0.984  
 &nbsp; &nbsp; Recall  &nbsp; &nbsp; &nbsp; 1.000  
 &nbsp; &nbsp; Specificity  &nbsp; &nbsp; &nbsp; 1.000
