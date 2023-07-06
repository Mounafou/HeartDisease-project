Heart Disease plan



"Heart Disease" dataset is taken from UCI Machine Learning Repository

https://archive.ics.uci.edu/ml/datasets/heart+disease
The data is converted into format accepted by sklearn.
The main goal is creating a prediction algorithm based on the 13 attributes outlied by the dataset.
The end result should have a GUI interface where the user can upload file or manually enter the attributes
and the program will give its prediction wheather or not it predict the 4 diseases for which the data is labeled

User manual
User is prompted by a GUI interface requesting to fill the patients data. The GIU interface lists 10 categorical values and 4 continous values. The 4 continous values only work with positive numbers or no input values and warns the user with popup sign in case of incorrect input. The program is capable to run with missing inputs. Naturally all missing inputs reduce prediction reliability.

Data handling
The machine learning weights are calculated using sklearn logistic regression hyperparameter tuned to work on the data. The weights are saved and transfered to the main part of the program.
The GUI interface loads the values selected by the user into a dictionary. Those values are filtered for incorrect user inputs. Converted to Pandas dataframe, on which the computer calculates the prediction based on the trainedweights on the input data.  

Created by Mounaf Anar and Dani
Mounaf was primarily responsible for creating and selecting the best machine learning methods and preparing the data.
Anar was primarily responsible for developing the GUI interface
Dani was primarily responsible for interconnecting the 2 previous codes
