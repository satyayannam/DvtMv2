import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd

# Declare global variables
df = pd.DataFrame()
columns = []

def load_dataset():
    global df, columns
    file_path = filedialog.askopenfilename(title="Select a file", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    if file_path:
        try:
            df = pd.read_csv(file_path)
            columns = df.columns.tolist()
            messagebox.showinfo("Success", "File loaded successfully!")
            column_dropdown.config(values=columns)  # Update column dropdown values
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file: {e}")

def perform_eda(eda_text, notebook):
    if not df.empty:
        eda_text.delete(1.0, tk.END)
        eda_text.insert(tk.END, f"Exploratory Data Analysis (EDA):\n\n")
        eda_text.insert(tk.END, f"Data Types:\n{df.dtypes}\n\n")
        eda_text.insert(tk.END, f"Missing Values:\n{df.isnull().sum()}\n\n")
        eda_text.insert(tk.END, f"Basic Statistics:\n{df.describe()}\n")
        eda_text.insert(tk.END, f"Correlation Matrix:\n{df.corr()}\n")
        notebook.select(0)  # Switch to the EDA screen tab (index 0)
    else:
        messagebox.showerror("Error", "No data loaded")


def clean_page(content_area):
    global df, columns, column_dropdown
    df = pd.DataFrame()  # Initialize an empty DataFrame

    # Clear previous content
    for widget in content_area.winfo_children():
        widget.destroy()

    # Load Data button
    load_button = ttk.Button(content_area, text="Load Dataset", command=load_dataset)
    load_button.pack(pady=10)

    # Create a scrolled text area for displaying EDA results
    eda_text = scrolledtext.ScrolledText(content_area, wrap=tk.WORD, width=80, height=20)
    eda_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Create a notebook for switching between different sections
    notebook = ttk.Notebook(content_area)
    notebook.pack(fill=tk.BOTH, expand=True)

    # Add a placeholder tab for the EDA screen
    eda_tab = ttk.Frame(notebook)
    notebook.add(eda_tab, text="EDA")

    # Perform EDA button
    perform_eda_button = ttk.Button(content_area, text="Perform EDA", command=lambda: perform_eda(eda_text, notebook))
    perform_eda_button.pack(pady=10)

    # Set the notebook to display the EDA screen initially
    notebook.select(eda_tab)

    # Dropdown menu to select column
    column_var = tk.StringVar(content_area)
    if columns:  # Check if columns list is not empty
        column_var.set(columns[0])  # Set default value
    else:
        column_var.set("")  # Set default value to empty string
    column_dropdown = ttk.Combobox(content_area, textvariable=column_var, values=columns)
    column_dropdown.pack(padx=10, pady=10)

    # Button to display selected column
    def display_selected_column(column_var):
        selected_column = column_var.get()
        if selected_column:
            selected_data = df[selected_column]
            selected_data_text.delete(1.0, tk.END)
            selected_data_text.insert(tk.END, str(selected_data))

    display_button = ttk.Button(content_area, text="Display Selected Column", command=lambda: display_selected_column(column_var))
    display_button.pack(pady=10)


    # Text area to display selected column's data
    selected_data_text = tk.Text(content_area, height=10, width=40)
    selected_data_text.pack(padx=10, pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Data Visualization Tool")
    root.geometry("800x600")

    # Create a content area
    content_area = ttk.Frame(root, width=800, height=600)
    content_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    clean_page(content_area)

    root.mainloop()
