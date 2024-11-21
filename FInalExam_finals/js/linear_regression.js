// Linear Regression Sample Data
const predictedScores = [85, 88, 90, 92, 95, 98, 100];
const actualScores = [80, 85, 88, 92, 97, 100, 103];

const ctx = document.getElementById('linearRegressionChart').getContext('2d');
const linearRegressionChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: ['Student 1', 'Student 2', 'Student 3', 'Student 4', 'Student 5', 'Student 6', 'Student 7'],
        datasets: [
            {
                label: 'Predicted Scores',
                data: predictedScores,
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                fill: true,
            },
            {
                label: 'Actual Scores',
                data: actualScores,
                borderColor: 'rgba(153, 102, 255, 1)',
                backgroundColor: 'rgba(153, 102, 255, 0.2)',
                fill: true,
            },
        ],
    },
    options: {
        scales: {
            y: {
                beginAtZero: true,
                max: 110,
            },
        },
    },
});
