import sys
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, render_template, request

PROJECT_PATH = os.path.dirname(os.path.abspath('C:\\Users\\user\\Documents\\codes'))  # Set the directory containing the CSV file
DATA_PATH = os.path.join(os.path.expanduser("~"), "Documents", "codes", "Online-Course-Recommendation-System", "udemy_courses.csv")
  # Adjust the path to your data file

sys.path.append(PROJECT_PATH)
#Course Recommendation Generator
class CourseGenerator:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.tf_idf_matrix = self.calculate_tfidf_matrix()

    def calculate_tfidf_matrix(self):
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.data['subject'])  # Use 'subject' column
        return tfidf_matrix

    def generate_recommendations(self, course_title, num_recommendations):
        # Initialize course_index to -1
        course_index = -1

        # Find the index of the course with the given title
        matching_courses = self.data[self.data['course_title'] == course_title]
        if not matching_courses.empty:
            course_index = matching_courses.index[0]

        if course_index >= 0:
            # Calculate the cosine similarity between the selected course and all other courses
            cosine_similarities = linear_kernel(self.tf_idf_matrix[course_index], self.tf_idf_matrix).flatten()

            # Get the indices of courses with the highest similarity scores
            related_course_indices = cosine_similarities.argsort()[:-num_recommendations-1:-1]

            # Exclude the selected course itself from recommendations
            related_course_indices = [i for i in related_course_indices if i != course_index]

            # Get the recommended course titles
            recommended_courses = self.data.iloc[related_course_indices]['course_title'].tolist()

            return recommended_courses
        else:
            return []
        

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    if request.method == "POST":
        # Get user input from the form
        course_title = request.form.get("course_title")
        num_recommendations = int(request.form.get("num_recommendations"))

        # Create an instance of CourseGenerator with the data path
        generator = CourseGenerator(DATA_PATH)

        # Generate course recommendations
        recommendations = generator.generate_recommendations(course_title, num_recommendations)

        # Render the result page with recommendations
        return render_template("result.html", course_title=course_title, recommendations=recommendations)
    else:
        return "Sorry, there was an error."

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
