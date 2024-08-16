# import tkinter as tk
# from tkinter import ttk, filedialog, messagebox, scrolledtext
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
#
# # Global variables
# df = pd.DataFrame()
#
# def load_dataset():
#     global df
#     file_path = filedialog.askopenfilename(title="Select a file", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
#     if file_path:
#         try:
#             df = pd.read_csv(file_path)
#             messagebox.showinfo("Success", "File loaded successfully!")
#         except Exception as e:
#             messagebox.showerror("Error", f"Error loading file: {e}")
#
# def mlops_page(content_area):
#     # Clear previous content
#     for widget in content_area.winfo_children():
#         widget.destroy()
#
#     # Load Data button
#     load_button = ttk.Button(content_area, text="Load Dataset", command=load_dataset)
#     load_button.pack(pady=10)
#
#     # Dropdown menu to select model
#     selected_model = tk.StringVar()
#     models_info = {
#         "Random Forest": {
#             "Model": RandomForestClassifier(),
#             "Description": "Random Forest algorithm is a powerful tree learning technique in Machine Learning."
#                            " It works by creating a number of Decision Trees during the training phase."
#                            " Each tree is constructed using a random subset of the data set to measure a random"
#                            " subset of features in each partition. This randomness introduces variability among individual trees,"
#                            " reducing the risk of overfitting and improving overall prediction performance. In prediction,"
#                            " the algorithm aggregates the results of all trees, either by voting (for classification tasks) "
#                            "or by averaging (for regression tasks) This collaborative decision-making process, supported by "
#                            "multiple trees with their insights, provides an example stable and precise results. ",
#             "Evaluation Scores": {
#                 "Accuracy": accuracy_score,
#                 "Precision": precision_score,
#                 "Recall": recall_score,
#                 "F1 Score": f1_score
#             },
#             "Score Formula": {
#                 "Accuracy": "(TP + TN) / (TP + TN + FP + FN)",
#                 "Precision": "TP / (TP + FP)",
#                 "Recall": "TP / (TP + FN)",
#                 "F1 Score": "2 * (Precision * Recall) / (Precision + Recall)"
#             }
#         },
#         "Decision Tree": {
#             "Model": DecisionTreeClassifier(),
#             "Description": "A decision tree is one of the most powerful tools of supervised learning algorithms"
#                            " used for both classification and regression tasks. It builds a flowchart-like tree "
#                            "structure where each internal node denotes a test on an attribute, each branch represents"
#                            " an outcome of the test, and each leaf node (terminal node) holds a class label. It is"
#                            " constructed by recursively splitting the training data into subsets based on the values"
#                            " of the attributes until a stopping criterion is met, such as the maximum depth of the tree"
#                            " or the minimum number of samples required to split a node.",
#             "Evaluation Scores": {
#                 "Accuracy": accuracy_score,
#                 "Precision": precision_score,
#                 "Recall": recall_score,
#                 "F1 Score": f1_score
#             },
#             "Score Formula": {
#                 "Accuracy": "(TP + TN) / (TP + TN + FP + FN)",
#                 "Precision": "TP / (TP + FP)",
#                 "Recall": "TP / (TP + FN)",
#                 "F1 Score": "2 * (Precision * Recall) / (Precision + Recall)"
#             }
#         },
#         "SVM": {
#             "Model": SVC(),
#             "Description": "Support Vector Machines (SVMs) are a type of supervised learning algorithm "
#                            "that can be used for classification or regression tasks. The main idea behind"
#                            " SVMs is to find a hyperplane that maximally separates the different classes in"
#                            " the training data. This is done by finding the hyperplane that has the largest "
#                            "margin, which is defined as the distance between the hyperplane and the closest "
#                            "data points from each class. Once the hyperplane is determined, new data can be "
#                            "classified by determining on which side of the hyperplane it falls. SVMs are particularly "
#                            "useful when the data has many features, and/or when there is a clear margin of separation in the data.",
#             "Evaluation Scores": {
#                 "Accuracy": accuracy_score,
#                 "Precision": precision_score,
#                 "Recall": recall_score,
#                 "F1 Score": f1_score
#             },
#             "Score Formula": {
#                 "Accuracy": "(TP + TN) / (TP + TN + FP + FN)",
#                 "Precision": "TP / (TP + FP)",
#                 "Recall": "TP / (TP + FN)",
#                 "F1 Score": "2 * (Precision * Recall) / (Precision + Recall)"
#             }
#         },
#         # Add more models and their evaluation scores as needed
#     }
#
#     model_dropdown = ttk.Combobox(content_area, textvariable=selected_model, values=list(models_info.keys()), state="readonly")
#     model_dropdown.pack(pady=10)
#
#     # Label to display model information and score formula
#     model_info_label = ttk.Label(content_area, text="", wraplength=400)
#     model_info_label.pack(pady=10)
#
#     # Function to update the label with the selected model's information
#     def update_model_info():
#         model_name = selected_model.get()
#         if model_name:
#             description = models_info[model_name]["Description"]
#             score_formula = models_info[model_name]["Score Formula"]
#             model_info_label.config(text=f"Description:\n{description}\n\nScore Formula:\n{score_formula}")
#
#     # Bind the model dropdown to the update_model_info function
#     model_dropdown.bind("<<ComboboxSelected>>", lambda event: update_model_info())
#
#     def make_prediction(features, model):
#         try:
#             input_data = []
#             prediction_window = tk.Toplevel()
#             prediction_window.title("Make Prediction")
#             prediction_window.geometry("300x200")
#
#             # Create a scrolled text area for input
#             input_text = scrolledtext.ScrolledText(prediction_window, wrap=tk.WORD, width=30, height=10)
#             input_text.grid(row=0, column=0, padx=10, pady=5)
#
#             def get_input():
#                 try:
#                     input_str = input_text.get(1.0, tk.END)
#                     input_data = [float(val) for val in input_str.split()]
#                     input_data = [input_data]  # Convert to 2D array
#                     prediction = model.predict(input_data)
#                     messagebox.showinfo("Prediction", f"The predicted value is: {prediction}")
#                     prediction_window.destroy()
#                 except ValueError as e:
#                     messagebox.showerror("Error", f"Invalid input: {e}")
#
#             predict_button = ttk.Button(prediction_window, text="Predict", command=get_input)
#             predict_button.grid(row=1, column=0, padx=10, pady=10)
#
#         except Exception as e:
#             messagebox.showerror("Error", f"An error occurred: {e}")
#
#     def predict_button_clicked():
#         if df.empty:
#             messagebox.showerror("Error", "No data loaded!")
#             return
#         target = df.iloc[:, -1]
#         features = df.iloc[:, :-1]
#         model_name = selected_model.get()
#         if model_name:
#             model = models_info[model_name]["Model"]
#             model.fit(features, target)
#             make_prediction(features.columns, model)
#
#     predict_button = ttk.Button(content_area, text="Predict", command=predict_button_clicked)
#     predict_button.pack(pady=10)
#
#     # Button to show evaluation scores
#     def show_scores():
#         if df.empty:
#             messagebox.showerror("Error", "No data loaded!")
#             return
#         target = df.iloc[:, -1]
#         features = df.iloc[:, :-1]
#         model_name = selected_model.get()
#         if model_name:
#             model = models_info[model_name]["Model"]
#             model.fit(features, target)
#             predictions = model.predict(features)
#             scores = {}
#             for score_name, score_func in models_info[model_name]["Evaluation Scores"].items():
#                 scores[score_name] = score_func(target, predictions)
#             score_texts = "\n".join([f"{score_name}: {score}" for score_name, score in scores.items()])
#             messagebox.showinfo("Evaluation Scores", score_texts)
#
#     scores_button = ttk.Button(content_area, text="Show Evaluation Scores", command=show_scores)
#     scores_button.pack(pady=10)
#
#     # Initially update the label with the first model's information
#     update_model_info()
#
#
# if __name__ == "__main__":
#     root = tk.Tk()
#     root.title("ML Operations")
#     root.geometry("800x600")
#
#     content_area = ttk.Frame(root, width=800, height=600)
#     content_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
#
#     mlops_page(content_area)
#
#     root.mainloop()





import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names")
def load_dataset():
    global df
    file_path = filedialog.askopenfilename(title="Select a file", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    if file_path:
        try:
            df = pd.read_csv(file_path)
            messagebox.showinfo("Success", "File loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file: {e}")

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

def mlops_page(content_area):
    # Clear previous content
    for widget in content_area.winfo_children():
        widget.destroy()

    # Load Data button
    load_button = ttk.Button(content_area, text="Load Dataset", command=load_dataset)
    load_button.pack(pady=10)

    def show_dataset():
        if df.empty:
            text_area.delete(1.0, tk.END)
            text_area.insert(tk.END, "Dataset is empty.")
        else:
            text_area.delete(1.0, tk.END)
            text_area.insert(tk.END, df.to_string())

    text_area = tk.Text(content_area, height=10, width=80)
    text_area.pack(pady=10)

    show_dataset_button = ttk.Button(content_area, text="Show Dataset", command=show_dataset)
    show_dataset_button.pack(pady=5)

    # Dropdown menu to select model
    selected_model = tk.StringVar()
    models_info = {
        "Random Forest": {
            "Model": RandomForestClassifier(),
            "Description": "Random Forest algorithm is a powerful tree learning technique in Machine Learning."
                           " It works by creating a number of Decision Trees during the training phase."
                           " Each tree is constructed using a random subset of the data set to measure a random"
                           " subset of features in each partition. This randomness introduces variability among individual trees,"
                           " reducing the risk of overfitting and improving overall prediction performance. In prediction,"
                           " the algorithm aggregates the results of all trees, either by voting (for classification tasks) "
                           "or by averaging (for regression tasks) This collaborative decision-making process, supported by "
                           "multiple trees with their insights, provides an example stable and precise results. ",
            "Evaluation Scores": {
                "Accuracy": accuracy_score,
                "Precision": precision_score,
                "Recall": recall_score,
                "F1 Score": f1_score
            },
            "Score Formula": {
                "Accuracy": "(TP + TN) / (TP + TN + FP + FN)",
                "Precision": "TP / (TP + FP)",
                "Recall": "TP / (TP + FN)",
                "F1 Score": "2 * (Precision * Recall) / (Precision + Recall)"
            }
        },
        "Decision Tree": {
            "Model": DecisionTreeClassifier(),
            "Description": "A decision tree is one of the most powerful tools of supervised learning algorithms"
                           " used for both classification and regression tasks. It builds a flowchart-like tree "
                           "structure where each internal node denotes a test on an attribute, each branch represents"
                           " an outcome of the test, and each leaf node (terminal node) holds a class label. It is"
                           " constructed by recursively splitting the training data into subsets based on the values"
                           " of the attributes until a stopping criterion is met, such as the maximum depth of the tree"
                           " or the minimum number of samples required to split a node.",
            "Evaluation Scores": {
                "Accuracy": accuracy_score,
                "Precision": precision_score,
                "Recall": recall_score,
                "F1 Score": f1_score
            },
            "Score Formula": {
                "Accuracy": "(TP + TN) / (TP + TN + FP + FN)",
                "Precision": "TP / (TP + FP)",
                "Recall": "TP / (TP + FN)",
                "F1 Score": "2 * (Precision * Recall) / (Precision + Recall)"
            }
        },
        "SVM": {
            "Model": SVC(),
            "Description": "Support Vector Machines (SVMs) are a type of supervised learning algorithm "
                           "that can be used for classification or regression tasks. The main idea behind"
                           " SVMs is to find a hyperplane that maximally separates the different classes in"
                           " the training data. This is done by finding the hyperplane that has the largest "
                           "margin, which is defined as the distance between the hyperplane and the closest "
                           "data points from each class. Once the hyperplane is determined, new data can be "
                           "classified by determining on which side of the hyperplane it falls. SVMs are particularly "
                           "useful when the data has many features, and/or when there is a clear margin of separation in the data.",
            "Evaluation Scores": {
                "Accuracy": accuracy_score,
                "Precision": precision_score,
                "Recall": recall_score,
                "F1 Score": f1_score
            },
            "Score Formula": {
                "Accuracy": "(TP + TN) / (TP + TN + FP + FN)",
                "Precision": "TP / (TP + FP)",
                "Recall": "TP / (TP + FN)",
                "F1 Score": "2 * (Precision * Recall) / (Precision + Recall)"
            }
        },
        # Add more models and their evaluation scores as needed
    }

    model_selection_label = ttk.Label(content_area, text="Select Model:")
    model_selection_label.pack(pady=5)

    model_dropdown = ttk.Combobox(content_area, textvariable=selected_model, values=list(models_info.keys()), state="readonly")
    model_dropdown.pack(pady=5)

    # Display model information when a model is selected
    def show_model_info():
        model_name = selected_model.get()
        model_info = models_info.get(model_name)
        if model_info:
            messagebox.showinfo("Model Information", model_info["Description"])

    show_info_button = ttk.Button(content_area, text="Show Model Info", command=show_model_info)
    show_info_button.pack(pady=5)

    def train_and_evaluate_model():
        model_name = selected_model.get()
        model_info = models_info.get(model_name)
        if model_info:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            X_train.fillna(0, inplace=True)  # Fill missing values with 0
            X_test.fillna(0, inplace=True)  # Fill missing values with 0

            # Convert data type if needed
            X_train = X_train.astype(float)
            X_test = X_test.astype(float)

            model = model_info["Model"]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            messagebox.showinfo("Model Evaluation Scores",
                                f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")

        else:
            messagebox.showerror("Error", "Model information not found")
            messagebox.showerror("Error", "Please select a model")

    evaluate_button = ttk.Button(content_area, text="Train and Evaluate Model", command=train_and_evaluate_model)
    evaluate_button.pack(pady=5)

    def make_prediction(features, model):


        try:
            input_data = []
            prediction_window = tk.Toplevel()
            prediction_window.title("Make Prediction")
            prediction_window.geometry("300x200")

            # Create a scrolled text area for input
            input_text = scrolledtext.ScrolledText(prediction_window, wrap=tk.WORD, width=30, height=10)
            input_text.grid(row=0, column=0, padx=10, pady=5)


            def get_input():
                try:
                    input_str = input_text.get(1.0, tk.END)
                    input_data = [float(val) for val in input_str.split()]
                    input_data = [input_data]  # Convert to 2D array
                    prediction = model.predict(input_data)
                    messagebox.showinfo("Prediction", f"The predicted value is: {prediction}")
                    prediction_window.destroy()
                except ValueError as e:
                    messagebox.showerror("Error", f"Invalid input: {e}")

            predict_button = ttk.Button(prediction_window, text="Predict", command=get_input)
            predict_button.grid(row=1, column=0, padx=10, pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def predict_button_clicked():
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        target = df.iloc[:, -1]
        features = df.iloc[:, :-1]
        model_name = selected_model.get()
        if model_name:
            model = models_info[model_name]["Model"]
            model.fit(features, target)
            make_prediction(features.columns, model)

    predict_button = ttk.Button(content_area, text="Predict", command=predict_button_clicked)
    predict_button.pack(pady=10)


if __name__ == "__main__":
    # Create the main window
    root = tk.Tk()
    root.title("ML Operations")
    root.geometry("800x600")

    # Create the content area
    content_area = ttk.Frame(root, width=800, height=600)
    content_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Call the mlops_page function to display the ML Ops page
    mlops_page(content_area)

    # Start the main event loop
    root.mainloop()