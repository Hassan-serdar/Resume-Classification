# &nbsp;Resume Classification using Scikit-learn \& TF-IDF \& NLP Fundamentals 

# 

# &nbsp;Project Overview

# 

# This project implements a machine learning pipeline to classify 962 resumes into one of 25 distinct job categories. The goal is to automate the initial screening process by accurately categorizing candidates (e.g., "Java Developer," "Testing," "DevOps Engineer") based on the text content of their resumes.

# 

# The final model, a Support Vector Classifier (SVC), achieved an accuracy of \*\*99.48%\*\* on the test set.

# 

# \##  Methodology \& Pipeline

# 

# The project follows these key steps, as detailed in the Jupyter Notebook:

# 

# 1\.  \*\*Data Loading \& Exploration:\*\*

# &nbsp;   \* Loaded the `UpdatedResumeDataSet.csv` (962 entries, 2 columns) using \*\*Pandas\*\*.

# &nbsp;   \* Analyzed the distribution of the 25 unique categories using \*\*Matplotlib (pie chart)\*\* and \*\*Seaborn (count plot)\*\*.

# 

# 2\.  \*\*Text Preprocessing:\*\*

# &nbsp;   \* Created a custom `cleanResume` function using \*\*Regex (re)\*\*.

# &nbsp;   \* This function removes URLs, mentions (`@`), hashtags (`#`), punctuation, non-ASCII characters, and extra whitespace from the raw resume text.

# 

# 3\.  \*\*Feature Engineering (Labels):\*\*

# &nbsp;   \* Converted the 25 text-based categories (e.g., "Data Science," "HR") into numerical labels (0-24) using `sklearn.preprocessing.LabelEncoder`.

# 

# 4\.  \*\*Feature Engineering (Text):\*\*

# &nbsp;   \* Transformed the cleaned resume text into a numerical matrix using `sklearn.feature\_extraction.text.TfidfVectorizer`, removing English stop words.

# 

# 5\.  \*\*Model Training \& Evaluation:\*\*

# &nbsp;   \* Split the data into training (80%) and testing (20%) sets.

# &nbsp;   \* Trained and evaluated two multiclass models using `sklearn.multiclass.OneVsRestClassifier`:

# &nbsp;       \* \*\*KNeighborsClassifier:\*\* Achieved 98.45% accuracy.

# &nbsp;       \* \*\*Support Vector Classifier (SVC):\*\* Achieved \*\*99.48% accuracy\*\* and was selected as the final model.

# &nbsp;   \* Evaluation metrics included Accuracy, Confusion Matrix, and a detailed Classification Report.

# 

# 6\.  \*\*Model Saving:\*\*

# &nbsp;   \* The trained `SVC` model, the `TfidfVectorizer`, and the `LabelEncoder` were saved to disk using `pickle` for future use (as `clf.pkl`, `tfidf.pkl`, and `encoder.pkl`).

# 

# \## Technologies \& Libraries Used

# 

# \* \*\*Language:\*\* Python

# \* \*\*Core Libraries:\*\* Pandas, NumPy

# \* \*\*Text Preprocessing:\*\* `re` (Regex)

# \* \*\*Data Visualization:\*\* Matplotlib, Seaborn

# \* \*\*Machine Learning:\*\* Scikit-learn

# &nbsp;   \* `TfidfVectorizer`

# &nbsp;   \* `LabelEncoder`

# &nbsp;   \* `train\_test\_split`

# &nbsp;   \* `OneVsRestClassifier`

# &nbsp;   \* `KNeighborsClassifier`

# &nbsp;   \* `SVC` (Support Vector Classifier)

# &nbsp;   \* `accuracy\_score`, `confusion\_matrix`, `classification\_report`

# \* \*\*Model Persistence:\*\* `pickle`

# \* \*\*Environment:\*\* Jupyter Notebook

# 

# \##  How to Run

# 

# 1\.  \*\*Clone the repository:\*\*

# &nbsp;   ```bash

# &nbsp;   git clone \[https://github.com/Hassan-serdar/Resume-Classification.git](https://github.com/Hassan-serdar/Resume-Classification.git)

# &nbsp;   cd Resume-Classification

# &nbsp;   ```

# 

# 2\.  \*\*Create a virtual environment:\*\*

# &nbsp;   ```bash

# &nbsp;   python -m venv venv

# &nbsp;   source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# &nbsp;   ```

# 

# 3\.  \*\*Install the required libraries:\*\*

# &nbsp;   \*Make sure to create a `requirements.txt` file first.\*

# &nbsp;   ```bash

# &nbsp;   pip install -r requirements.txt

# &nbsp;   ```

# 

# 4\.  \*\*Run the Jupyter Notebook:\*\*

# &nbsp;   ```bash

# &nbsp;   jupyter notebook "Resume Classification.ipynb"

# &nbsp;   ```

# &nbsp;   You can run the cells sequentially to see the full analysis, training, and prediction process.

