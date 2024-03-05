import sys
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from joblib import dump, load
import pandas as pd
import matplotlib.pyplot as plt
import coremltools


features = ['angles', 'points', 'normalized_points']
classifiers = ['svc', 'ert', 'gb', 'lr']
exercise_class_ids = {'push-up': 0, 'pull-up': 1, 'squat': 2}
stage_class_ids = {'start': 0, 'end': 1}

def prepare_dataset(ds_source, class_ids, reduce_to_min=False):
    print('Gathering data from CSV...')
    data = pd.read_csv(ds_source)
    # exclude data points with missing variables
    data = data.dropna()
    
    if (reduce_to_min):
        sample_counts = [data["class"].value_counts()[class_name] for class_name in class_ids]
        smallest_count = min(sample_counts)
        for class_name in class_ids:
            diff = data["class"].value_counts()[class_name] - smallest_count
            #print(data[data['class'] == class_name].sample(diff))
            data = data.drop(data[data['class'] == class_name].sample(diff, random_state=1).index)
            
    print(data.head())

    # Separate ground truth labels and data values 
    input_data = data.drop(columns=['class'])
    labels = data['class']

    return input_data, labels

def compose_datasets(ds_source, class_ids, test_ds_source = None):
    print('Preparing training data...')
    X_train, y_train = prepare_dataset(ds_source, class_ids)
    if (test_ds_source):
        print('Preparing test data...')
        X_test, y_test = prepare_dataset(test_ds_source, class_ids, True)
    else:
        print('Splitting dataset...')
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=38)

    return [X_train, y_train], [X_test, y_test]

def train_classifier(train_data, classifier_type, balance_weights=True):
    X, y = train_data
    print('Training classifier...')
    if classifier_type == 'svc':
        classifier = SVC(class_weight='balanced' if balance_weights else None)
    elif classifier_type == 'ert':
        classifier = ExtraTreesClassifier(random_state=1)
    elif classifier_type == 'gb':
        classifier = GradientBoostingClassifier(n_estimators=100, random_state=1)
    else:
        classifier = LogisticRegression(max_iter=200)
    classifier.fit(X, y)

    return classifier

def test_classifier(classifier, test_data, class_ids):
    X, y = test_data
    predictions = classifier.predict(X)
    predicted_accuracy = accuracy_score(y, predictions)
    print(predicted_accuracy)

    # Generate confusion matrix for predictions
    predicted_matrix = confusion_matrix(y, predictions, normalize='true')
    disp = ConfusionMatrixDisplay(predicted_matrix, display_labels=class_ids.keys())
    disp.plot()
    plt.show()

def export_classifier(classifier, target_path):
    print('Exporting classifier...')
    # Pickled model
    dump(classifier, target_path + '.joblib')
    # CoreML model
    coreml_model = coremltools.converters.sklearn.convert(classifier, 'pose_angles', 'predicted_stage')
    coreml_model.save(target_path + '.mlmodel')
    print('Export finished.')


if (__name__ == '__main__'):
    # mode and feature
    try:
        if sys.argv[1] in classifiers and sys.argv[2] in features:
            classifier_type = sys.argv[1]
            feature = sys.argv[2]

            if len(sys.argv) > 3 and sys.argv[3] in exercise_class_ids:
				# stage dataset
                mode = 'stage_'
                exercise = '/' + sys.argv[3]
                class_ids = stage_class_ids
            else: 
                mode = ''
                exercise = ''	
                class_ids = exercise_class_ids	
        else:
            raise Exception()
    except:
        print('Classifier or feature not permitted. Please try again.')
        exit(0)
        
    train_dataset_path = f'../../../BA/{mode}training_data{exercise}/{mode}training_{feature}.csv'
    test_dataset_path = f'../../../BA/{mode}test_data{exercise}/{mode}test_{feature}.csv'
    model_target_path = f'./models/{classifier_type}_{feature}{exercise.replace("/", "_").replace("-", "")}_{mode}classifier'
          
    train_data, test_data = compose_datasets(train_dataset_path, class_ids, test_dataset_path)
    action_svc = train_classifier(train_data, classifier_type)

    test_classifier(action_svc, test_data, class_ids)
    export_classifier(action_svc, model_target_path)
