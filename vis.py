
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox

import mplcursors
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings

# Suppress mplcursors warning about missing pick support for PolyCollection
warnings.filterwarnings("ignore", message="Pick support for PolyCollection is missing.")

df = pd.DataFrame()

def load_dataset():
    global df
    file_path = filedialog.askopenfilename(title="Select a file")
    if file_path:
        df = pd.read_csv(file_path)
        update_feature_labels()

def update_feature_labels():
    global features_dropdown, labels_dropdown
    if not df.empty:
        features = list(df.columns)
        features_dropdown['values'] = features
        labels_dropdown['values'] = features

def plot_graph():
    global features_dropdown, labels_dropdown, plot_type_var, plot_frame
    if not df.empty:
        feature = features_dropdown.get()
        label = labels_dropdown.get()
        plot_type = plot_type_var.get()

        if feature and label and plot_type:
            try:
                # Clear previous plot
                for widget in plot_frame.winfo_children():
                    widget.destroy()

                fig, ax = plt.subplots(figsize=(8, 6))
                if plot_type == "Scatter Plot":
                    ax.scatter(df[feature], df[label])
                elif plot_type == "Bar Chart":
                    ax.bar(df[feature], df[label])
                elif plot_type == "Box Plot":
                    ax.boxplot(df[feature])
                elif plot_type == "Line Plot":
                    for col in df.columns:
                        ax.plot(df[col], label=col)
                    ax.legend()
                elif plot_type == "Histogram":
                    ax.hist(df[feature])
                elif plot_type == "Violin Plot":
                    ax.violinplot(df[feature])
                else:
                    messagebox.showerror("Error", "Invalid plot type selected")
                    return
                ax.set_xlabel(feature)
                ax.set_ylabel(label)
                ax.set_title(plot_type)

                # Add zoomable and scalable features
                mplcursors.cursor(ax, hover=True).connect("add", lambda sel: sel.annotation.set_text(f"{sel.artist.get_label()}: {sel.target[1]}"))

                canvas = FigureCanvasTkAgg(fig, master=plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

                # Configure the scrollbars
                vbar = tk.Scrollbar(plot_frame, orient=tk.VERTICAL, command=canvas.get_tk_widget().yview)
                vbar.pack(side=tk.RIGHT, fill=tk.Y)
                hbar = tk.Scrollbar(plot_frame, orient=tk.HORIZONTAL, command=canvas.get_tk_widget().xview)
                hbar.pack(side=tk.BOTTOM, fill=tk.X)
                canvas.get_tk_widget().config(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

            except Exception as e:
                messagebox.showerror("Error", str(e))
        else:
            messagebox.showerror("Error", "Select feature, label, and plot type")
    else:
        messagebox.showerror("Error", "No data loaded")


def on_closing():
    root.destroy()

def visualization_page(content_area):
    for widget in content_area.winfo_children():
        widget.destroy()

    load_button = ttk.Button(content_area, text="Load Dataset", command=load_dataset)
    load_button.pack(pady=10)

    global features_dropdown, labels_dropdown, plot_type_var, plot_frame, notes_text
    features_label = ttk.Label(content_area, text="Select Features:")
    features_label.pack()
    features_dropdown = ttk.Combobox(content_area, values=[], state="readonly")
    features_dropdown.pack()

    labels_label = ttk.Label(content_area, text="Select Labels:")
    labels_label.pack()
    labels_dropdown = ttk.Combobox(content_area, values=[], state="readonly")
    labels_dropdown.pack()

    plot_type_label = ttk.Label(content_area, text="Select Plot Type:")
    plot_type_label.pack()
    plot_type_var = tk.StringVar()
    plot_type_dropdown = ttk.Combobox(content_area, values=["Scatter Plot", "Bar Chart", "Box Plot", "Line Plot", "Histogram","Violin Plot"], state="readonly", textvariable=plot_type_var)
    plot_type_dropdown.pack()

    show_plot_button = ttk.Button(content_area, text="Show Plot", command=plot_graph)
    show_plot_button.pack(pady=10)

    # Create a frame to contain the plot and notes area
    plot_notes_frame = tk.Frame(content_area)
    plot_notes_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    plot_frame = tk.Frame(plot_notes_frame, bg="#ccffff", width=600, height=400)
    plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    notes_label = ttk.Label(plot_notes_frame, text="Write Notes:")
    notes_label.pack()
    notes_text = scrolledtext.ScrolledText(plot_notes_frame, height=10, width=40)
    notes_text.pack(pady=5)

    content_area.grid_rowconfigure(0, weight=1)
    content_area.grid_columnconfigure(1, weight=1)

    notes_text.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)  # Pack the notes area to the right

def on_closing():
    plt.close('all')  # Close all matplotlib plot windows
    root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("ML Operations")
    root.geometry("1000x800")

    content_area = ttk.Frame(root, width=800, height=600)
    content_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    root.protocol("WM_DELETE_WINDOW", on_closing)  # Handle the window closing event
    visualization_page(content_area)

    root.mainloop()