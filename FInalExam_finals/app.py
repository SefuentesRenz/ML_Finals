from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/linear_regression', methods=['GET', 'POST'])
def linear_regression():
    prediction = None
    error_message = None
    if request.method == 'POST':
        try:
            department = request.form['department']
            prev_enrollment_year_1 = int(request.form['prev_enrollment_year_1'])
            prev_enrollment_year_2 = int(request.form['prev_enrollment_year_2'])
            prev_enrollment_year_3 = int(request.form['prev_enrollment_year_3'])
        except ValueError:
            error_message = "Invalid input. Please enter numeric values for all fields."
            return render_template('linear_regression.html', error_message=error_message)

        # Sample historical enrollment data (you can replace this with actual data)
        data = pd.DataFrame({
            'Department': ['Computer Science', 'Information Technology', 'Mechanical Engineering', 'Civil Engineering', 'Biology'],
            'PrevYear1': [100, 120, 90, 110, 80],
            'PrevYear2': [105, 125, 95, 115, 85],
            'PrevYear3': [110, 130, 100, 120, 90],  
            'EnrolmentNextYear': [115, 135, 105, 125, 95]
        })

        # Filter data for the selected department
        department_data = data[data['Department'] == department]
        
        if department_data.empty:
            error_message = "Department not found."
            return render_template('linear_regression.html', error_message=error_message)

        # Prepare the data for linear regression
        X = department_data[['PrevYear1', 'PrevYear2', 'PrevYear3']]
        y = department_data['EnrolmentNextYear']

        model = LinearRegression()
        model.fit(X, y)

        # Predict the enrollment for the next year based on user input
        prediction = model.predict([[prev_enrollment_year_1, prev_enrollment_year_2, prev_enrollment_year_3]])[0]

    return render_template('linear_regression.html', prediction=prediction, error_message=error_message)
        


@app.route('/naive_bayes', methods=['GET', 'POST'])
def predict_professor():
    prediction = None
    error_message = None

    if request.method == 'POST':
        try:
            # Get form data
            engagement = float(request.form['engagement'])
            clarity = float(request.form['clarity'])
            knowledge = float(request.form['knowledge'])
            communication = float(request.form['communication'])
            overall_satisfaction = float(request.form['overall_satisfaction'])

        except ValueError:
            error_message = "Invalid input. Please enter numeric values for all fields."
            return render_template('predict_professor.html', error_message=error_message)

        # Example training data: student evaluations of professors
        data = pd.DataFrame({
            'Engagement': [85, 70, 95, 80, 65],
            'Clarity': [90, 75, 80, 85, 70],
            'Knowledge': [88, 80, 85, 78, 65],
            'Communication': [92, 70, 88, 84, 68],
            'OverallSatisfaction': [90, 70, 95, 85, 60],
            'Rating': [4.5, 3.2, 4.8, 4.1, 3.0]  # Professor ratings (out of 5)
        })

        # Prepare data for prediction
        X = data[['Engagement', 'Clarity', 'Knowledge', 'Communication', 'OverallSatisfaction']]
        y = data['Rating']

        # Train a simple linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Make prediction based on input data
        input_data = [[engagement, clarity, knowledge, communication, overall_satisfaction]]
        numeric_prediction = model.predict(input_data)[0]

        # Convert the numeric prediction into a category
        if numeric_prediction >= 4.5:
            prediction = 'Excellent'
        elif numeric_prediction >= 3.5:
            prediction = 'Good'
        elif numeric_prediction >= 2.5:
            prediction = 'Neutral'
        elif numeric_prediction >= 1.5:
            prediction = 'Bad'
        else:
            prediction = 'Poor'

    return render_template('naive_bayes.html', prediction=prediction, error_message=error_message)


@app.route('/knn', methods=['GET', 'POST'])
def knn():
    prediction = None
    error_message = None
    if request.method == 'POST':
        try:
            # Retrieve form data
            department = request.form['department']
            prev_enrollment_year_1 = int(request.form['prev_enrollment_year_1'])
            prev_enrollment_year_2 = int(request.form['prev_enrollment_year_2'])
            prev_enrollment_year_3 = int(request.form['prev_enrollment_year_3'])
        except ValueError:
            error_message = "Invalid input. Please enter numeric values for previous enrollment years."
            return render_template('knn.html', error_message=error_message)

        # Sample historical enrollment data
        data = pd.DataFrame({
            'Department': ['Computer Science', 'Information Technology', 'Mechanical Engineering', 'Civil Engineering', 'Biology'],
            'PrevYear1': [100, 120, 90, 110, 80],
            'PrevYear2': [105, 125, 95, 115, 85],
            'PrevYear3': [110, 130, 100, 120, 90],
            'EnrolmentNextYear': [115, 135, 105, 125, 95]
        })

        # Filter data for the selected department (case-insensitive match)
        department_data = data[data['Department'].str.lower() == department.lower()]

        if department_data.empty:
            error_message = f"No data found for department: {department}. Please select a valid department."
            return render_template('knn.html', error_message=error_message)

        # Prepare the data for KNN regression
        X = department_data[['PrevYear1', 'PrevYear2', 'PrevYear3']]
        y = department_data['EnrolmentNextYear']

        # Initialize and train KNN Regressor
        try:
            model = KNeighborsRegressor(n_neighbors=1)  # Adjust n_neighbors as needed
            model.fit(X, y)

            # Predict the enrollment for the next year
            prediction = model.predict([[prev_enrollment_year_1, prev_enrollment_year_2, prev_enrollment_year_3]])[0]
        except Exception as e:
            error_message = f"An error occurred during prediction: {str(e)}"
            return render_template('knn.html', error_message=error_message)

    return render_template('knn.html', prediction=prediction, error_message=error_message)

@app.route('/svm', methods=['GET', 'POST'])
def svm():
    prediction = None
    error_message = None

    if request.method == 'POST':
        try:
            # Get form data
            engagement = float(request.form['engagement'])
            clarity = float(request.form['clarity'])
            knowledge = float(request.form['knowledge'])
            communication = float(request.form['communication'])
            overall_satisfaction = float(request.form['overall_satisfaction'])
        except ValueError:
            error_message = "Invalid input. Please enter numeric values for all fields."
            return render_template('svm.html', error_message=error_message)

        # Example dataset: student evaluations of professors
        data = pd.DataFrame({
            'Engagement': [85, 70, 95, 80, 65],
            'Clarity': [90, 75, 80, 85, 70],
            'Knowledge': [88, 80, 85, 78, 65],
            'Communication': [92, 70, 88, 84, 68],
            'OverallSatisfaction': [90, 70, 95, 85, 60],
            'Rating': [4.5, 3.2, 4.8, 4.1, 3.0]  # Professor ratings (out of 5)
        })

        # Prepare data for prediction
        X = data[['Engagement', 'Clarity', 'Knowledge', 'Communication', 'OverallSatisfaction']]
        y = data['Rating']

        # Initialize and train the Support Vector Regression model
        model = SVR(kernel='linear')  # Use SVR instead of SVC for regression
        model.fit(X, y)

        # Prepare the input data for prediction
        input_data = [[engagement, clarity, knowledge, communication, overall_satisfaction]]
        
        # Make prediction
        prediction_value = model.predict(input_data)[0]

        # Convert numeric prediction into a category
        if prediction_value >= 4.5:
            prediction = 'Excellent'
        elif prediction_value >= 3.5:
            prediction = 'Good'
        elif prediction_value >= 2.5:
            prediction = 'Neutral'
        elif prediction_value >= 1.5:
            prediction = 'Bad'
        else:
            prediction = 'Poor'

    return render_template('svm.html', prediction=prediction, error_message=error_message)


@app.route('/decision_tree', methods=['GET', 'POST'])
def decision_tree():
    prediction = None
    error_message = None

    if request.method == 'POST':
        try:
            # Get form data and ensure they are filled out
            engagement = float(request.form['engagement'])
            clarity = float(request.form['clarity'])
            knowledge = float(request.form['knowledge'])
            communication = float(request.form['communication'])
            overall_satisfaction = float(request.form['overall_satisfaction'])
        except ValueError as e:
            error_message = f"Invalid input. Please enter numeric values for all fields. Error: {str(e)}"
            return render_template('decision_tree.html', error_message=error_message)

        try:
            # Example training data
            data = pd.DataFrame({
                'Engagement': [85, 70, 95, 80, 65],
                'Clarity': [90, 75, 80, 85, 70],
                'Knowledge': [88, 80, 85, 78, 65],
                'Communication': [92, 70, 88, 84, 68],
                'OverallSatisfaction': [90, 70, 95, 85, 60],
                'Rating': [4.5, 3.2, 4.8, 4.1, 3.0]  # Ratings
            })

            # Prepare data for prediction
            X = data[['Engagement', 'Clarity', 'Knowledge', 'Communication', 'OverallSatisfaction']]
            y = data['Rating']

            # Initialize and train the Decision Tree Regressor
            model = DecisionTreeRegressor()  # Use Regressor, not Classifier
            model.fit(X, y)

            # Prepare the input data for prediction
            input_data = [[engagement, clarity, knowledge, communication, overall_satisfaction]]

            # Make prediction
            numeric_prediction = model.predict(input_data)[0]

            # Convert the numeric prediction into a category
            if numeric_prediction >= 4.5:
                prediction = 'Excellent'
            elif numeric_prediction >= 3.5:
                prediction = 'Good'
            elif numeric_prediction >= 2.5:
                prediction = 'Neutral'
            elif numeric_prediction >= 1.5:
                prediction = 'Bad'
            else:
                prediction = 'Poor'
        except Exception as e:
            error_message = f"An error occurred during prediction: {str(e)}"
            return render_template('decision_tree.html', error_message=error_message)

    return render_template('decision_tree.html', prediction=prediction, error_message=error_message)



@app.route('/ann', methods=['GET', 'POST'])
def neural_network():
    prediction = None
    error_message = None

    if request.method == 'POST':
        try:
            # Get form data and validate if they are float values
            engagement = float(request.form['engagement'])
            clarity = float(request.form['clarity'])
            knowledge = float(request.form['knowledge'])
            communication = float(request.form['communication'])
            overall_satisfaction = float(request.form['overall_satisfaction'])
        except ValueError:
            error_message = "Invalid input. Please enter numeric values for all fields."
            return render_template('ann.html', error_message=error_message)

        try:
            # Example dataset for ANN training
            data = pd.DataFrame({
                'Engagement': [85, 70, 95, 80, 65],
                'Clarity': [90, 75, 80, 85, 70],
                'Knowledge': [88, 80, 85, 78, 65],
                'Communication': [92, 70, 88, 84, 68],
                'OverallSatisfaction': [90, 70, 95, 85, 60],
                'Rating': [4.5, 3.2, 4.8, 4.1, 3.0]  # Ratings out of 5
            })

            # Encode ratings to integer categories
            label_encoder = LabelEncoder()
            data['Rating'] = label_encoder.fit_transform(data['Rating'])

            # Prepare the data
            X = data[['Engagement', 'Clarity', 'Knowledge', 'Communication', 'OverallSatisfaction']]
            y = data['Rating']

            # Train ANN model
            model = MLPClassifier(max_iter=500)
            model.fit(X, y)

            # Make prediction
            input_data = [[engagement, clarity, knowledge, communication, overall_satisfaction]]
            prediction_value = model.predict(input_data)[0]

            # Decode the prediction back to the original category
            prediction = label_encoder.inverse_transform([prediction_value])[0]
            if prediction_value >= 4.5:
                prediction = 'Excellent'
            elif prediction_value >= 3.5:
                prediction = 'Good'
            elif prediction_value >= 2.5:
                prediction = 'Average'
            else:
                prediction = 'Poor'

        except Exception as e:
            error_message = f"An error occurred during prediction: {str(e)}"
            return render_template('ann.html', error_message=error_message)

    return render_template('ann.html', prediction=prediction, error_message=error_message)
if __name__ == '__main__':
    app.run(debug=True)
