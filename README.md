# mapa-evaluation

Usage:

mapa_evaluation.py path/to/directory level

Ouput example:

Called function: class_boundary_evaluation.....................................
=================== Error Types ===================
			Level 1		Level 2		Total
Boundary		 19 		 0 		 19
Class			 15 		 5 		 20
Boundary and Class	 0 		 3 		 3
Total			 34 		 8 		 42

=================== Class Errors ===================
Level 1:
	 PERSON
		 ORGANISATION 			 6
	 ORGANISATION
		 ADDRESS 			 8
		 PERSON 			 1

Level 2:
	 given name - female
		 given name - male 		 2
		 family name 			 2
	 postcode
		 street 			 1
	 standard abbreviation
		 day 				 1
		 month 				 1
		 year 				 1

Called function: metrics_per_level.............................................
Computing at LEVEL 1 and LEVEL 2 globally
Scores at LEVEL 1
	 F-score 				 0.951
	 Precision 				 0.958
	 Recall 				 0.945
	 Specificity 				 0.997

Scores at LEVEL 2
	 F-score 				 0.969
	 Precision 				 0.993
	 Recall 				 0.946
	 Specificity 				 1.000

Called function: metrics_per_category..........................................
Computing at LEVEL 1 and LEVEL 2
Scores at LEVEL1
	 ADDRESS
		 F-score 			 0.950
		 Precision 			 0.915
		 Recall 			 0.989
		 Specificity 			 0.999
	 AMOUNT
		 F-score 			 0.390
		 Precision 			 0.667
		 Recall 			 0.276
		 Specificity 			 1.000
	 DATE
		 F-score 			 0.978
		 Precision 			 0.997
		 Recall 			 0.959
		 Specificity 			 1.000
	 ORGANISATION
		 F-score 			 0.794
		 Precision 			 0.757
		 Recall 			 0.835
		 Specificity 			 0.997
	 PERSON
		 F-score 			 0.991
		 Precision 			 0.997
		 Recall 			 0.984
		 Specificity 			 1.000
	 TIME
		 F-score 			 0
		 Precision 			 0
		 Recall 			 0
		 Specificity 			 0

Scores at LEVEL 2
	 ROLE
		 F-score 			 1.000
		 Precision 			 1.000
		 Recall 			 1.000
		 Specificity 			 1.000
	 city
		 F-score 			 0.969
		 Precision 			 0.939
		 Recall 			 1.000
		 Specificity 			 1.000
	 day
		 F-score 			 0.995
		 Precision 			 0.991
		 Recall 			 1.000
		 Specificity 			 1.000
	 family name
		 F-score 			 0.974
		 Precision 			 0.983
		 Recall 			 0.966
		 Specificity 			 1.000
	 given name - female
		 F-score 			 0
		 Precision 			 0
		 Recall 			 0
		 Specificity 			 0
	 given name - male
		 F-score 			 0.917
		 Precision 			 0.846
		 Recall 			 1.000
		 Specificity 			 1.000
	 month
		 F-score 			 0.995
		 Precision 			 0.990
		 Recall 			 1.000
		 Specificity 			 1.000
	 postcode
		 F-score 			 0.952
		 Precision 			 1.000
		 Recall 			 0.909
		 Specificity 			 1.000
	 standard abbreviation
		 F-score 			 0.812
		 Precision 			 1.000
		 Recall 			 0.683
		 Specificity 			 1.000
	 street
		 F-score 			 0.978
		 Precision 			 0.978
		 Recall 			 0.978
		 Specificity 			 1.000
	 title
		 F-score 			 0.996
		 Precision 			 1.000
		 Recall 			 0.993
		 Specificity 			 1.000
	 unit
		 F-score 			 0.333
		 Precision 			 0.750
		 Recall 			 0.214
		 Specificity 			 1.000
	 value
		 F-score 			 0.400
		 Precision 			 0.800
		 Recall 			 0.267
		 Specificity 			 1.000
	 year
		 F-score 			 0.992
		 Precision 			 0.984
		 Recall 			 1.000
		 Specificity 			 1.000
