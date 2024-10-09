import tkinter as tk
from drawingapp import DrawingApp

def main():
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()