// Naive Bayes Classification Data
const feedbackData = [70, 30]; // 70% Positive, 30% Negative

const ctx = document.getElementById('naiveBayesChart').getContext('2d');
const naiveBayesChart = new Chart(ctx, {
    type: 'pie',
    data: {
        labels: ['Positive Feedback', 'Negative Feedback'],
        datasets: [{
            data: feedbackData,
            backgroundColor: ['rgba(75, 192, 192, 1)', 'rgba(255, 99, 132, 1)'],
            hoverBackgroundColor: ['rgba(75, 192, 192, 0.8)', 'rgba(255, 99, 132, 0.8)'],
        }],
    },
});
