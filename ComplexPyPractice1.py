"""This is a test"""

import tkinter as tk
from tkinter import messagebox, PhotoImage
import numpy as np

# Splash screen function
def show_splash_screen():
    splash = tk.Toplevel()
    splash.title("Welcome")
    splash.geometry("400x300")
    splash.overrideredirect(True)
    tk.Label(splash, text="Welcome to Physics Learning Model", font=("Arial", 16, "bold")).pack(pady=20)
    tk.Label(splash, text="Washington & Lee University", font=("Arial", 14)).pack()
    splash_logo = PhotoImage(file="school.png")  # Replace with your school logo file
    tk.Label(splash, image=splash_logo).pack(pady=20)
    
    def close_splash():
        splash.destroy()
    splash.after(3000, close_splash)  # Show for 3 seconds
    root.wait_window(splash)

# Define the functions for the programs
# (Existing calculators retained with better formatting)

def calculate_kinetic_energy():
    def compute():
        try:
            mass = float(mass_entry.get())
            velocity = float(velocity_entry.get())
            if mass < 0 or velocity < 0:
                raise ValueError("Mass and velocity must be non-negative.")
            ke = 0.5 * mass * velocity ** 2
            result_label.config(text=f"Kinetic Energy: {ke:.2f} J")
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))

    window = tk.Toplevel(root)
    window.title("Kinetic Energy Calculator")

    tk.Label(window, text="Calculate the energy of a moving object:", font=("Arial", 12)).pack(pady=10)

    tk.Label(window, text="Mass (kg):").pack()
    mass_entry = tk.Entry(window)
    mass_entry.pack()

    tk.Label(window, text="Velocity (m/s):").pack()
    velocity_entry = tk.Entry(window)
    velocity_entry.pack()

    tk.Button(window, text="Calculate", command=compute).pack(pady=10)
    result_label = tk.Label(window, text="", font=("Arial", 12))
    result_label.pack()

# Additional calculators (falling distance, current, projectile range) follow the same pattern as above

# Main GUI setup
root = tk.Tk()
root.title("Physics Learning Model")
root.geometry("600x400")

show_splash_screen()

frame = tk.Frame(root)
frame.pack(pady=20)

header = tk.Label(frame, text="Physics Learning Model", font=("Arial", 16, "bold"))
header.pack()

subheader = tk.Label(frame, text="Choose a calculator to explore physics concepts", font=("Arial", 12))
subheader.pack(pady=10)

programs = [
    ("Kinetic Energy Calculator", calculate_kinetic_energy),
    # Other calculators here (e.g., Falling Distance, Current, etc.)
]

for name, func in programs:
    tk.Button(frame, text=name, command=func, font=("Arial", 12), bg="#4CAF50", fg="white", padx=10, pady=5).pack(fill="x", pady=5)

footer = tk.Label(root, text="Developed by Abid & Jonathan", font=("Arial", 10, "italic"))
footer.pack(side="bottom", pady=10)

root.mainloop()
