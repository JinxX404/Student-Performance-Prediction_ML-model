# Import necessary libraries
import joblib
from ttkthemes import ThemedTk
from tkinter import ttk
from tkinter import messagebox

# Create the root window with a theme
root = ThemedTk(theme="arc")
root.title("Student Performance Prediction System")

# Set the window size
window_width = 800
window_height = 600
root.geometry(f"{window_width}x{window_height}")

# Create a frame in the window
frame = ttk.Frame(root,padding=150)
frame.pack(padx=10, pady=10)

# Create labels and entry fields for each feature
failures_label = ttk.Label(frame, text="Number of past class failures:")
failures_label.pack()
failures_entry = ttk.Entry(frame)
failures_entry.pack()

absences_label = ttk.Label(frame, text="Number of school absences (numeric: from 0 to 93):")
absences_label.pack()
absences_entry = ttk.Entry(frame)
absences_entry.pack()

G1_label = ttk.Label(frame, text="First period grade (numeric: from 0 to 20):")
G1_label.pack()
G1_entry = ttk.Entry(frame)
G1_entry.pack()

G2_label = ttk.Label(frame, text="Second period grade (numeric: from 0 to 20)")
G2_label.pack()
G2_entry = ttk.Entry(frame)
G2_entry.pack()

# Load the model
model = joblib.load('student_performance_model.joblib')

# Function to make prediction
def make_prediction():
    try:
        # Get the input from the user
        failures = float(failures_entry.get())
        absences = float(absences_entry.get())
        G1 = float(G1_entry.get())
        G2 = float(G2_entry.get())
        G2_G1 = G2-G1
        # Make the prediction
        prediction = model.predict([[failures,absences,G1,G2,G2_G1]])
        # Show the prediction in a message box
        messagebox.showinfo("Final grade", prediction[0])
    except:
        messagebox.showinfo("Error",'Please enter a number')

# Create the predict button
button = ttk.Button(frame, text="Predict", command=make_prediction)
button.pack(padx=(10, 0), pady=(10, 0))

# Start the main loop
root.mainloop()