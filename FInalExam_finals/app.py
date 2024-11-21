from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

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


if __name__ == '__main__':
    app.run(debug=True)



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
            department = request.form['department']
            prev_enrollment_year_1 = int(request.form['prev_enrollment_year_1'])
            prev_enrollment_year_2 = int(request.form['prev_enrollment_year_2'])
            prev_enrollment_year_3 = int(request.form['prev_enrollment_year_3'])
        except ValueError:
            error_message = "Invalid input. Please enter numeric values for all fields."
            return render_template('knn.html', error_message=error_message)

        # Sample historical enrollment data
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
            return render_template('knn.html', error_message=error_message)

        # Prepare the data for KNN regression
        X = department_data[['PrevYear1', 'PrevYear2', 'PrevYear3']]
        y = department_data['EnrolmentNextYear']

        # Initialize KNN Regressor
        model = KNeighborsRegressor(n_neighbors=2)  # n_neighbors can be adjusted based on data
        model.fit(X, y)

        # Predict the enrollment for the next year based on user input
        prediction = model.predict([[prev_enrollment_year_1, prev_enrollment_year_2, prev_enrollment_year_3]])[0]

    return render_template('knn.html', prediction=prediction, error_message=error_message)

@app.route('/svm', methods=['GET', 'POST'])
def svm():
    if request.method == 'POST':
        time_taken = float(request.form['time_taken'])
        answers_correct = float(request.form['answers_correct'])

        data = pd.DataFrame({
            'TimeTaken': [30, 45, 20, 60, 35],
            'AnswersCorrect': [15, 18, 10, 25, 20],
            'Cheating': [0, 0, 1, 0, 1]
        })
        X = data[['TimeTaken', 'AnswersCorrect']]
        y = data['Cheating']

        model = SVC(kernel='linear')
        model.fit(X, y)
        prediction = model.predict([[time_taken, answers_correct]])
        return f"Cheating Detected: {'Yes' if prediction[0] else 'No'}"
    return '''
        <form method="post">
            Time Taken: <input type="text" name="time_taken"><br>
            Answers Correct: <input type="text" name="answers_correct"><br>
            <input type="submit" value="Check Cheating">
        </form>
    '''

@app.route('/decision-tree', methods=['GET', 'POST'])
def decision_tree():
    if request.method == 'POST':
        diagnostic_score = float(request.form['diagnostic_score'])
        preferred_pace = int(request.form['preferred_pace'])

        data = pd.DataFrame({
            'DiagnosticScore': [85, 70, 65, 90, 75],
            'PreferredPace': [1, 0, 0, 1, 0],
            'NextTopic': ['Advanced', 'Intermediate', 'Beginner', 'Advanced', 'Intermediate']
        })
        X = data[['DiagnosticScore', 'PreferredPace']]
        y = data['NextTopic']

        model = DecisionTreeClassifier()
        model.fit(X, y)
        prediction = model.predict([[diagnostic_score, preferred_pace]])
        return f"Next Recommended Topic: {prediction[0]}"
    return '''
        <form method="post">
            Diagnostic Score: <input type="text" name="diagnostic_score"><br>
            Preferred Pace (1/0): <input type="text" name="preferred_pace"><br>
            <input type="submit" value="Find Topic">
        </form>
    '''

@app.route('/neural-network', methods=['GET', 'POST'])
def neural_network():
    if request.method == 'POST':
        engagement = float(request.form['engagement'])
        grades = float(request.form['grades'])

        data = pd.DataFrame({
            'Engagement': [80, 50, 70, 30, 90],
            'Grades': [85, 60, 75, 40, 95],
            'Dropout': [0, 1, 0, 1, 0]
        })
        X = data[['Engagement', 'Grades']]
        y = data['Dropout']

        model = MLPClassifier(max_iter=500)
        model.fit(X, y)
        prediction = model.predict([[engagement, grades]])
        return f"Dropout Risk: {'High' if prediction[0] else 'Low'}"
    return '''
        <form method="post">
            Engagement: <input type="text" name="engagement"><br>
            Grades: <input type="text" name="grades"><br>
            <input type="submit" value="Check Risk">
        </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
