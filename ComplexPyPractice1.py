import tkinter as tk
from tkinter import messagebox
import numpy as np

# Define the functions for the programs
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

    tk.Label(window, text="Mass (kg):").grid(row=0, column=0)
    mass_entry = tk.Entry(window)
    mass_entry.grid(row=0, column=1)

    tk.Label(window, text="Velocity (m/s):").grid(row=1, column=0)
    velocity_entry = tk.Entry(window)
    velocity_entry.grid(row=1, column=1)

    tk.Button(window, text="Calculate", command=compute).grid(row=2, columnspan=2)
    result_label = tk.Label(window, text="")
    result_label.grid(row=3, columnspan=2)

def calculate_falling_distance():
    def compute():
        try:
            g = 9.8
            output = "Time (s)\tDistance (m)\n" + "-" * 20 + "\n"
            for t in range(1, 11):
                distance = 0.5 * g * t ** 2
                output += f"{t}\t\t{distance:.2f}\n"
            result_text.config(state="normal")
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, output)
            result_text.config(state="disabled")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    window = tk.Toplevel(root)
    window.title("Falling Distance Calculator")

    tk.Button(window, text="Calculate", command=compute).pack()
    result_text = tk.Text(window, height=15, width=40, state="disabled")
    result_text.pack()

def calculate_current():
    def compute():
        try:
            voltage = float(voltage_entry.get())
            resistance = float(resistance_entry.get())
            if resistance <= 0:
                raise ValueError("Resistance must be positive.")
            current = voltage / resistance
            result_label.config(text=f"Current: {current:.2f} A")
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))

    window = tk.Toplevel(root)
    window.title("Current Calculator")

    tk.Label(window, text="Voltage (V):").grid(row=0, column=0)
    voltage_entry = tk.Entry(window)
    voltage_entry.grid(row=0, column=1)

    tk.Label(window, text="Resistance (Ω):").grid(row=1, column=0)
    resistance_entry = tk.Entry(window)
    resistance_entry.grid(row=1, column=1)

    tk.Button(window, text="Calculate", command=compute).grid(row=2, columnspan=2)
    result_label = tk.Label(window, text="")
    result_label.grid(row=3, columnspan=2)

def calculate_projectile_range():
    def compute():
        try:
            g = 9.8
            v0 = float(velocity_entry.get())
            angles = np.arange(10, 91, 10)
            output = "Angle (°)\tRange (m)\n" + "-" * 20 + "\n"
            for angle in angles:
                angle_rad = np.radians(angle)
                range_distance = (v0 ** 2) * np.sin(2 * angle_rad) / g
                output += f"{angle}\t\t{range_distance:.2f}\n"
            result_text.config(state="normal")
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, output)
            result_text.config(state="disabled")
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))

    window = tk.Toplevel(root)
    window.title("Projectile Range Calculator")

    tk.Label(window, text="Initial Velocity (m/s):").pack()
    velocity_entry = tk.Entry(window)
    velocity_entry.pack()

    tk.Button(window, text="Calculate", command=compute).pack()
    result_text = tk.Text(window, height=15, width=40, state="disabled")
    result_text.pack()

# Main GUI setup
root = tk.Tk()
root.title("Program Selector")

programs = [
    ("Kinetic Energy Calculator", calculate_kinetic_energy),
    ("Falling Distance Calculator", calculate_falling_distance),
    ("Current Calculator", calculate_current),
    ("Projectile Range Calculator", calculate_projectile_range),
]

tk.Label(root, text="Select a program:").pack()

for name, func in programs:
    tk.Button(root, text=name, command=func).pack(fill="x", padx=10, pady=5)

root.mainloop()
