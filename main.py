# import tkinter as tk
# import webbrowser
# from tkinter import ttk
#
# from matplotlib import pyplot as plt
#
# import clean
# import mlops
# import vis
#
#
# class MainApp(tk.Tk):
#     def open_email(self):
#         webbrowser.open_new("mailto:xensindo@gmail.com")
#
#     def __init__(self):
#         super().__init__()
#         self.title("Home")
#         self.geometry("800x600")
#         self.minsize(height=800, width=800)
#         self.style = ttk.Style()
#         self.style.configure("Custom.TFrame", background="#b3b3ff")
#
#         # Create a side menu bar
#         self.side_menu = ttk.Frame(self, width=600, height=600, style='Custom.TFrame')
#         self.side_menu.pack(side=tk.LEFT, fill=tk.Y)
#
#         # Create buttons in the side menu
#         self.button1 = ttk.Button(self.side_menu, text="Clean", command=self.clean_page)
#         self.button1.pack(pady=10)
#         self.button2 = ttk.Button(self.side_menu, text="ML ops", command=self.mlops_page)
#         self.button2.pack(pady=10)
#         self.button3 = ttk.Button(self.side_menu, text="Visualize", command=self.vis_page)
#         self.button3.pack(pady=10)
#
#         # Create a content area
#         self.content_area = ttk.Frame(self, width=1200, height=1200)
#         self.content_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
#
#         # Instructions label
#         self.instructions = ttk.Label(self.content_area, text="Data visualization tool using machine learning",
#                                        justify="left", font=("AngsanaUPC", 8))
#         self.instructions.pack(expand=True)
#         self.label = ttk.Label(self.content_area, text="Have a question? Click below to email us:")
#         self.label.pack(pady=10)
#
#         self.button = ttk.Button(self.content_area, text="Email Us", command=self.open_email)
#         self.button.pack(pady=10)
#
#     def clean_page(self):
#         clean.clean_page(self.content_area)
#
#     def mlops_page(self):
#         mlops.mlops_page(self.content_area)  # Pass the df argument
#
#     def vis_page(self):
#         vis.visualization_page(self.content_area)  # Placeholder for visualization page
#
# def on_closing():
#     plt.close('all')  # Close all matplotlib plot windows
#     root.destroy()
#
# if __name__ == "__main__":
#     root = MainApp()
#     root.protocol("WM_DELETE_WINDOW", on_closing)  # Handle the window closing event
#     root.mainloop()
#



import tkinter as tk
from tkinter import ttk
from tkinter import PhotoImage

from matplotlib import pyplot as plt

import clean
import mlops
import vis

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Home")
        self.geometry("800x600")
        self.minsize(height=800, width=800)
        self.style = ttk.Style()
        self.style.configure("Custom.TFrame", background="#FFFFFF")
        self.style.configure("main.TFrame", background="#DEC6E3")
        # Create a side menu bar
        self.side_menu = ttk.Frame(self, width=150, height=600,style="Custom.TFrame")
        self.side_menu.pack(side=tk.LEFT, fill=tk.Y)

        # Add icons (you need to have your own image files)
        clean_icon = PhotoImage(file='an.png').subsample(2, 2)
        mlops_icon = PhotoImage(file='ml.png').subsample(2, 2)
        visualize_icon = PhotoImage(file='vis.png').subsample(2, 2)

        # Create buttons in the side menu
        self.button1 = ttk.Button(self.side_menu, image=clean_icon, compound=tk.LEFT, command=self.clean_page, style='SideMenu.TButton')
        self.button1.image = clean_icon
        self.button1.pack(pady=10)
        self.button2 = ttk.Button(self.side_menu, image=mlops_icon, compound=tk.LEFT, command=self.mlops_page, style='SideMenu.TButton')
        self.button2.image = mlops_icon
        self.button2.pack(pady=10)
        self.button3 = ttk.Button(self.side_menu, image=visualize_icon, compound=tk.LEFT, command=self.vis_page, style='SideMenu.TButton')
        self.button3.image = visualize_icon
        self.button3.pack(pady=10)

        # Create a content area
        self.content_area = ttk.Frame(self, width=650, height=600,style="main.TFrame")
        self.content_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Instructions label
        self.instructions = ttk.Label(self.content_area, text="Data visualization tool using machine learning v2",
                                       justify="left", font=("AngsanaUPC", 8))
        self.instructions.pack(expand=True)
        self.label = ttk.Label(self.content_area, text="Have a question? Click below to email us:")
        self.label.pack(pady=10)
    def clean_page(self):
        clean.clean_page(self.content_area)

    def mlops_page(self):
        mlops.mlops_page(self.content_area)  # Pass the df argument

    def vis_page(self):
        vis.visualization_page(self.content_area)  # Placeholder for visualization page
#
def on_closing():
    plt.close('all')  # Close all matplotlib plot windows
    root.destroy()

if __name__ == "__main__":
    root = MainApp()
    root.protocol("WM_DELETE_WINDOW", on_closing)  # Handle the window closing event
    root.mainloop()
