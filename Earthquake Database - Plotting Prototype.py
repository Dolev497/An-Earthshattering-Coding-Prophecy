import sqlite3 as sq
import matplotlib.pyplot as pyplot
import sklearn.linear_model
import sklearn.neighbors
import sklearn.metrics
import sklearn.model_selection
import numpy 
import copy
from sklearn.model_selection import GridSearchCV
def make_table(columns, rows, values, fig_name):
    table = pyplot.table(rowLabels = rows,
                    colLabels = columns,
                    cellText = values,
                    cellLoc = 'center',
                    loc = 'center')
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 2.5)
    pyplot.axis('off')
    pyplot.savefig(fig_name, bbox_inches="tight")
class Earthquake_Database:
    def __init__(self, file, dictionary_path):
        '''Initializing defense dictionary, database'''
        #Creating the dictionary to later convert string description to numerical value
        with open(dictionary_path, "r", encoding="UTF-8") as dic_file:
            self.defense_dictionary = {} 
            for records in dic_file:
                temp = records.split(",")
                value = temp[-1]
                self.defense_dictionary[temp[0]] = int(round(float(value[0:-1])))
        #Establishing the database (either from a csv/text file, or a .db database file)
        extension = file.split(".")
        self.connection  = sq.connect(f"{extension[0]}.db")
        self.cur = self.connection.cursor()
        if extension[1] != "db":
            self.buildings = []
            self.cur.execute("CREATE TABLE Earthquakes(earthquake_id INT AUTO_INCREMENT PRIMARY KEY, Defense_level string, magnitude int, density int, damage_level int, longlat text, location text, distance integer)")
            with open(file, "r", encoding="UTF-8") as text_file:
                for records in text_file:
                        temp = records.split(',')
                        for a in range(int(temp[-1])):
                            self.buildings.append(list(temp[0:-1]))
            self.cur.executemany('INSERT INTO Earthquakes (defense_level, magnitude, density, damage_level, longlat, location, distance) VALUES (?, ?, ?, ?, ?, ?, ?)', self.buildings)
            self.connection.commit()
        #Converting data in database to feature vectors and labels    
        self.Feature_Vectors = []
        self.Label_Sequence = []
        self.Distinct_Feature_Vectors = []
        self.Distinct_Label_Sequence = []
        for rows in self.cur.execute("SELECT defense_level, magnitude, density, distance, damage_level FROM Earthquakes"):
                self.Feature_Vectors.append(list(rows[0:-1]))
                if rows[-1] > 3:
                    self.Label_Sequence.append(1)
                else:
                    self.Label_Sequence.append(0)
        for rows in self.cur.execute("SELECT DISTINCT defense_level, magnitude, density, distance, damage_level FROM Earthquakes"):
                self.Distinct_Feature_Vectors.append(list(rows[0:-1]))
                if rows[-1] > 3:
                    self.Distinct_Label_Sequence.append(1)
                else:
                    self.Distinct_Label_Sequence.append(0)
    
    def check_defense_dictionary(self, vectors):
        '''Recieves a feature vector with an unconverted defense level, checks dictionary and returns feature vector with numerical defense value.
        If defense level not found in the dictionary, raises an error.'''
        temp_vectors = copy.deepcopy(vectors)
        if type(vectors[0]) == type([]):
            for count, vector in enumerate(vectors):
                if vector[0] in self.defense_dictionary:
                    temp_vector = copy.copy(vector)
                    temp_vector[0] = self.defense_dictionary[vector[0]]
                    temp_vectors[count] = temp_vector
                else:
                    raise FileNotFoundError
            return temp_vectors
        else:
            if vectors[0] in self.defense_dictionary:
                    temp_vectors[0] = self.defense_dictionary[vectors[0]]
                    return temp_vectors
            else:
                raise FileNotFoundError
    def load_test_vectors(self, file):
        self.test_feature_vectors = []
        try:
            self.cur.execute("CREATE TABLE Test_Earthquakes(earthquake_id INT AUTO_INCREMENT PRIMARY KEY, Defense_level string, magnitude int, density int, location text, distance integer, district text)")
            self.test_buildings = []
            with open(file, "r", encoding="UTF-8") as text_file:
                for records in text_file:
                        temp = records.split(',')
                        for a in range(int(temp[-1])):
                            self.test_buildings.append(list(temp[0:-1]))
            self.cur.executemany('INSERT INTO Test_Earthquakes (defense_level, magnitude, density, location, distance, district) VALUES (?, ?, ?, ?, ?, ?)', self.test_buildings)
            self.connection.commit()
        except sq.OperationalError:
            pass
        for rows in self.cur.execute("SELECT defense_level, magnitude, density, distance FROM Test_Earthquakes"):
                self.test_feature_vectors.append(list(rows))
    
    def get_test_feature_vectors(self):
        return self.test_feature_vectors
    def get_feature_vectors(self):
        '''Returns feature vectors'''
        return self.Feature_Vectors
    def get_labels(self):
        '''Returns label sequence'''
        return self.Label_Sequence
    def get_distinct_feature_vectors(self):
        '''Returns distinct feature vectors'''
        return self.Distinct_Feature_Vectors
    def get_distinct_labels(self):
        '''Returns label sequence for distinct vectors'''
        return self.Distinct_Label_Sequence
    def run_sql_query(self, query, variable_tuple = None):
        '''Runs an SQL Query, returns selection.'''
        if variable_tuple == None:
            return self.cur.execute(query)
        return self.cur.execute(query, variable_tuple)
    
    def get_feats_labels_from_query(self, query, variable_tuple = None):
        '''Runs SQL selection query, returns feature vectors and labels of selection.'''
        Feature_Vectors = []
        Label_Sequence = []
        if variable_tuple == None:
            for rows in self.cur.execute(query):
                Feature_Vectors.append(list(rows[0:-1]))
                if rows[-1] > 3:
                    Label_Sequence.append(1)
                else:
                    Label_Sequence.append(0)
                return Feature_Vectors, Label_Sequence
        else:
            for rows in self.cur.execute(query, variable_tuple):
                Feature_Vectors.append(list(rows[0:-1]))
                if rows[-1] > 3:
                    Label_Sequence.append(1)
                else:
                    Label_Sequence.append(0)
            return Feature_Vectors, Label_Sequence
    '''def plot_all(self):
        features = numpy.array(self.Feature_Vectors)
        labels = numpy.array(self.Label_Sequence)
        pass
'''
class LR_Model:
    '''Logistic Regression Class'''
    def __init__(self, train_vectors, train_labels):
        '''Creates model and fits to train feature vectors'''
        self.model = sklearn.linear_model.LogisticRegression(class_weight={0:276, 1:2000}).fit(train_vectors, train_labels)
        self.coefficients = self.model.coef_
        self.statistics = {}
    def run_TP_FP_TN_FN(self, vectors, labels, probability_threshold=0.6):
        "Runs model on test data and returns accuracy statistics"
        probabilities = self.model.predict_proba(vectors)
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for count, probability in enumerate(probabilities):
            if probability[1] >= probability_threshold:
                if labels[count] == 1:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if labels[count] == 0:
                    true_negative += 1
                else:
                    false_negative += 1
        return true_positive, false_positive, true_negative, false_negative
    def run_prediction_testing(self, all_vectors, all_labels, number_of_splits, weights_of_classes = {0:276, 1:2000}):
        average_accuracy_score = 0
        average_area_under_curve = 0
        average_recall_score = 0
        for splits in range(number_of_splits):
            vector_train, vector_test, label_train, label_test = sklearn.model_selection.train_test_split(all_vectors, all_labels, test_size=0.2)
            temp_model = sklearn.linear_model.LogisticRegression(class_weight=weights_of_classes).fit(vector_train, label_train)
            average_accuracy_score += sklearn.metrics.accuracy_score(label_test, temp_model.predict(vector_test))
            average_area_under_curve += sklearn.metrics.roc_auc_score(label_test, temp_model.predict(vector_test))
            average_recall_score += sklearn.metrics.recall_score(label_test, temp_model.predict(vector_test))
        average_accuracy_score /= number_of_splits
        average_area_under_curve /= number_of_splits
        average_recall_score /= number_of_splits
        return average_accuracy_score, average_area_under_curve, average_recall_score
    def predict_from_test_vector(self, test_vectors, district):
        fallen_buildings = 0
        standing_buildings = 0
        percent_sum = 0
        for a in self.model.predict_proba(test_vectors):
            percent_sum += a[1]
            if a[1] > 0.75:
                fallen_buildings+= 1
            else:
                standing_buildings += 1
        total_samples = fallen_buildings+standing_buildings
        self.statistics[district] = [fallen_buildings, standing_buildings, total_samples, round(fallen_buildings/total_samples, 3), round(percent_sum/total_samples, 3)]
    def get_statistics(self):
        return self.statistics   
class KNN:
    '''K Nearest Neighbors Algorithm'''
    def __init__(self, feature_vectors, labels):
        '''Establishes distance between each feature vector and stores it in a distance dictionary'''
        self.train_vectors = feature_vectors
        self.labels = labels
        ''' 
        k_range = list(range(1, 31))
        weight_options = ['uniform', 'distance']
        metric_options = ['seuclidean', 'manhattan', 'p', 'canberra', 'l2', 'haversine', 'braycurtis', 'sokalmichener', 'cosine', 'minkowski', 'l1', 'hamming', 'euclidean', 'infinity', 'yule', 'kulsinski', 'matching', 'correlation', 'nan_euclidean', 'russellrao', 'mahalanobis', 'sqeuclidean', 'chebyshev', 'jaccard', 'pyfunc', 
            'rogerstanimoto', 'sokalsneath', 'cityblock', 'dice', 'wminkowski']
        self.param_grid = dict(n_neighbors=k_range, weights=weight_options, metric=metric_options)
        self.KNN_Model = GridSearchCV(estimator=sklearn.neighbors.KNeighborsClassifier(), param_grid= self.param_grid, verbose=1, cv=10, n_jobs=-1, scoring="recall").fit(self.train_vectors, self.labels)
        print(self.KNN_Model.best_score_)
        print(self.KNN_Model.best_params_)
        '''
        self.KNN_Model  = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights="distance", metric = "cosine").fit(self.train_vectors, self.labels)
        
        self.statistics = {}
        '''
        self.distance_dictionary = {}
        for feature_vector in feature_vectors:
            self.distance_dictionary[feature_vector] = {}
            for also_feature_vector in feature_vectors:
                if also_feature_vector != feature_vector:
                    self.distance_dictionary[feature_vector][also_feature_vector] = minkowskiDist(feature_vector, also_feature_vector)
        '''
    def run_test_KNN(self, test_vectors, labels, probability_threshold):
        probabilities = self.KNN_Model.predict_proba(test_vectors)
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for count, probability in enumerate(probabilities):
            if probability[1] >= probability_threshold:
                if labels[count] == 1:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if labels[count] == 0:
                    true_negative += 1
                else:
                    false_negative += 1
        return true_positive, false_positive, true_negative, false_negative
    def predict_from_test_vector(self, test_vectors, district):
        fallen_buildings = 0
        standing_buildings = 0
        percent_sum = 0
        for a in self.KNN_Model.predict_proba(test_vectors):
            percent_sum += a[1]
            if a[1] > 0.4:
                fallen_buildings+= 1
            else:
                standing_buildings += 1
        total_samples = fallen_buildings+standing_buildings
        self.statistics[district] = [fallen_buildings, standing_buildings, total_samples, round(fallen_buildings/total_samples, 3), round(percent_sum/total_samples, 3)]
    def run_prediction_testing(self, all_vectors, all_labels, number_of_splits, weights_of_classes = {0:276, 1:2000}):
        average_accuracy_score = 0
        average_area_under_curve = 0
        average_recall_score = 0
        for splits in range(number_of_splits):
            vector_train, vector_test, label_train, label_test = sklearn.model_selection.train_test_split(all_vectors, all_labels, test_size=0.2)
            temp_model = sklearn.linear_model.LogisticRegression(class_weight=weights_of_classes).fit(vector_train, label_train)
            average_accuracy_score += sklearn.metrics.accuracy_score(label_test, temp_model.predict(vector_test))
            average_area_under_curve += sklearn.metrics.roc_auc_score(label_test, temp_model.predict(vector_test))
            average_recall_score += sklearn.metrics.recall_score(label_test, temp_model.predict(vector_test))
        average_accuracy_score /= number_of_splits
        average_area_under_curve /= number_of_splits
        average_recall_score /= number_of_splits
        return average_accuracy_score, average_area_under_curve, average_recall_score
    def get_statistics(self):
        return self.statistics   
#Establishing Database    
Proto_run  = Earthquake_Database("Near_Done_Model/Near Complete Model.db", "Near_Done_Model/Unadjusted Defense Dictionary.csv")

#Establishing Test Data

Proto_run.load_test_vectors("Near_Done_Model/Earthquake Database Prototype Dataset - Israel 2022.csv")

test_vectors = Proto_run.check_defense_dictionary(Proto_run.get_test_feature_vectors())
#Creating+Fitting Model
Logistic_Regression_Model = LR_Model(Proto_run.check_defense_dictionary(Proto_run.get_feature_vectors()), Proto_run.get_labels())
KNN_Algorithm = KNN(Proto_run.check_defense_dictionary(Proto_run.get_distinct_feature_vectors()), Proto_run.get_distinct_labels())

#Run Model for Statistics
#Logistic Regression Stats
LR_stats = Logistic_Regression_Model.run_TP_FP_TN_FN(Proto_run.check_defense_dictionary(Proto_run.get_feature_vectors()), Proto_run.get_labels(), 0.6)
print(f"True Positives {LR_stats[0]}, False Positives {LR_stats[1]}, True Negatives {LR_stats[2]}, False Negatives {LR_stats[3]}")
predicted_labels = Logistic_Regression_Model.model.predict(Proto_run.check_defense_dictionary(Proto_run.get_feature_vectors()))
print(f"Accuracy Score: {sklearn.metrics.accuracy_score(Proto_run.get_labels(), predicted_labels)}, Area Under Curve = {sklearn.metrics.roc_auc_score(Proto_run.get_labels(), predicted_labels)}, Recall Score = {sklearn.metrics.recall_score(Proto_run.get_labels(), predicted_labels)}")
accuracy, area_under_curve, recall = Logistic_Regression_Model.run_prediction_testing(Proto_run.check_defense_dictionary(Proto_run.get_feature_vectors()), Proto_run.get_labels(), 15)
print(f"Results of Logistic Regression Train-Test-Split = Accuracy Score = {accuracy}, Area Under Curve = {area_under_curve}, Recall Score = {recall}")
#KNN Stats
KNN_stats = KNN_Algorithm.run_test_KNN(Proto_run.check_defense_dictionary(Proto_run.get_distinct_feature_vectors()), Proto_run.get_distinct_labels(), 0.3)
print(f"True Positives {KNN_stats[0]}, False Positives {KNN_stats[1]}, True Negatives {KNN_stats[2]}, False Negatives {KNN_stats[3]}")
predicted_labels_KNN = KNN_Algorithm.KNN_Model.predict(Proto_run.check_defense_dictionary(Proto_run.get_distinct_feature_vectors()))
print(f"Accuracy Score: {sklearn.metrics.accuracy_score(Proto_run.get_distinct_labels(), predicted_labels_KNN)}, Area Under Curve = {sklearn.metrics.roc_auc_score(Proto_run.get_distinct_labels(), predicted_labels_KNN)}, Recall Score = {sklearn.metrics.recall_score(Proto_run.get_distinct_labels(), predicted_labels_KNN)}")
KNN_accuracy, KNN_area_under_curve, KNN_recall = KNN_Algorithm.run_prediction_testing(Proto_run.check_defense_dictionary(Proto_run.get_feature_vectors()), Proto_run.get_labels(), 15)
print(f"Results of KNN Train-Test-Split = Accuracy Score = {KNN_accuracy}, Area Under Curve = {KNN_area_under_curve}, Recall Score = {KNN_recall}")
#Establish Sub-Districts
sub_districts = ["Jerusalem", "Beer Sheva", "Ashkelon", "Golan", "Zfat", "Kinneret", "Akko", "Yizre'el", "Haifa", "Hadera", "Sharon", "Petah Tikvah", "Ramla", "Rehovot", "Tel Aviv/Jaffa", "Judea and Samaria"]

#Get Prediction Statistics For Logistic Regression Model And KNN Model
for district in sub_districts:
    Israel_Feature_Vectors = []
    for rows in Proto_run.run_sql_query("SELECT defense_level, magnitude, density, distance FROM Test_Earthquakes WHERE District = (?)", (district,)):
        Israel_Feature_Vectors.append(list(rows))
    Logistic_Regression_Model.predict_from_test_vector(Proto_run.check_defense_dictionary(Israel_Feature_Vectors), district)
    KNN_Algorithm.predict_from_test_vector(Proto_run.check_defense_dictionary(Israel_Feature_Vectors), district)
KNN_predictions = KNN_Algorithm.get_statistics()
logistic_regression_predictions = Logistic_Regression_Model.get_statistics()

print(f"The KNN predictions are: {KNN_predictions.items()}")
print(f"The Logistic Regression predictions are: {logistic_regression_predictions.items()}")
        
'''
minkowskidists = []
densities = []
print(Proto_run.get_distinct_feature_vectors())
for vector in Proto_run.get_distinct_feature_vectors():
    density = 0
    distance = minkowskiDist([0,0,0,0], Proto_run.check_defense_dictionary(vector), 1)
    print(distance)
    for row in Proto_run.run_sql_query("SELECT * FROM Earthquakes WHERE defense_level = (?) AND magnitude = (?) AND density = (?) AND distance = (?)", vector):
        density += 1
    minkowskidists.append(distance)
    densities.append(density)
minkowskidists = numpy.array(minkowskidists)
densities = numpy.array(densities)
pyplot.figure()
pyplot.plot(minkowskidists, Proto_run.get_distinct_labels(), "ro")
pyplot.title("Distance From 0 Vector")
pyplot.show()
'''
#Finding label ratio
fallen = 0
notfallen = 0
for a in Proto_run.run_sql_query("SELECT damage_level FROM Earthquakes"):
    if a[0] > 3:
        fallen += 1
    else:
        notfallen += 1
print(f"Not Fallen to fallen ratio is notfallen:fallen = {notfallen}:{fallen}")

#Making a Results Table
make_table(["No. Fallen", "No. Not Fallen", "Sample Size", "Destruction %", "Confidence Avg"],
            sub_districts, list(logistic_regression_predictions.values()), 'Logistic Regression Results')
make_table(["No. Fallen", "No. Not Fallen", "Sample Size", "Destruction %", "Confidence Avg"],
            sub_districts, list(KNN_predictions.values()), 'K-Nearest Neighbors Results')
Proto_run.connection.commit()
Proto_run.connection.close()    

