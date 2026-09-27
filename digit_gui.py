import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tensorflow.keras.models import load_model

# -------------------------------
# 1. Load the trained CNN model
# -------------------------------
model = load_model("mnist_cnn_model.h5")  # make sure this file is in the same folder

# -------------------------------
# 2. Create main GUI window
# -------------------------------
window = tk.Tk()
window.title("Handwritten Digit Recognition")

# Canvas dimensions
canvas_width = 200
canvas_height = 200

# -------------------------------
# 3. Create drawing canvas
# -------------------------------
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="white")
canvas.grid(row=0, column=0, columnspan=4)

# Create PIL image to capture drawing
image1 = Image.new("L", (canvas_width, canvas_height), color=255)
draw = ImageDraw.Draw(image1)

# -------------------------------
# 4. Drawing functions
# -------------------------------
def paint(event):
    # Draw on canvas
    x1, y1 = (event.x - 8), (event.y - 8)
    x2, y2 = (event.x + 8), (event.y + 8)
    canvas.create_oval(x1, y1, x2, y2, fill='black', width=0)
    draw.ellipse([x1, y1, x2, y2], fill='black')

canvas.bind("<B1-Motion>", paint)  # B1 = left mouse button

# -------------------------------
# 5. Clear canvas
# -------------------------------
def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill=255)
    label.config(text="Draw a digit and click Predict")

# -------------------------------
# 6. Predict digit
# -------------------------------
def predict_digit():
    # Resize to 28x28 (MNIST format)
    img = image1.resize((28,28))
    img = ImageOps.invert(img)  # invert colors (white background → black digits)
    img = np.array(img)/255.0
    img = img.reshape(1,28,28,1)
    
    # Make prediction
    pred = model.predict(img)
    digit = np.argmax(pred)
    
    # Show result
    label.config(text=f"Predicted Digit: {digit}")

# -------------------------------
# 7. Buttons
# -------------------------------
tk.Button(window, text="Predict", command=predict_digit, width=10, bg="lightgreen").grid(row=1, column=0)
tk.Button(window, text="Clear", command=clear, width=10, bg="lightcoral").grid(row=1, column=1)

# -------------------------------
# 8. Label to display prediction
# -------------------------------
label = tk.Label(window, text="Draw a digit and click Predict", font=("Helvetica", 14))
label.grid(row=2, column=0, columnspan=4)

# -------------------------------
# 9. Start the GUI
# -------------------------------
window.mainloop()