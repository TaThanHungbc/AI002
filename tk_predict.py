# tk_predict.py (updated)
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import torch
from predict import predict_and_explain  # predict_and_explain(model_path, image_path, device)

MODEL_PATH = "best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def open_and_predict():
    path = filedialog.askopenfilename(filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.tiff")])
    if not path:
        return

    # show thumbnail
    try:
        img = Image.open(path)
        img.thumbnail((600,600))
        imgtk = ImageTk.PhotoImage(img)
        img_label.config(image=imgtk)
        img_label.image = imgtk
    except Exception as e:
        messagebox.showerror("Error", f"Không thể mở ảnh: {e}")
        return

    # update status
    status_var.set("Processing... please wait")
    root.update_idletasks()

    try:
        final_label, explanations = predict_and_explain(MODEL_PATH, path, device)
    except Exception as e:
        status_var.set("Ready")
        messagebox.showerror("Error", f"Cannot predict: {e}")
        return

    # display label (AI or Human) in bigger font and color
    if final_label == "AI":
        label_text = "AI"
        label_color = "red"
        label_full = "AI-generated"
    else:
        label_text = "Human"
        label_color = "green"
        label_full = "Human-written"

    result_label.config(text=label_full, fg=label_color)

    # clear and insert explanations (one per line)
    text_box.configure(state='normal')
    text_box.delete("1.0", tk.END)
    for ex in explanations:
        text_box.insert(tk.END, "• " + ex + "\n\n")
    text_box.configure(state='disabled')

    status_var.set("Ready")

# GUI setup
root = tk.Tk()
root.title("Handwriting Classifier (Human vs AI)")

top_frame = tk.Frame(root)
top_frame.pack(padx=8, pady=8, fill='x')

btn = tk.Button(top_frame, text="Open image and predict", command=open_and_predict)
btn.pack(side='left')

status_var = tk.StringVar(value="Ready")
status_label = tk.Label(top_frame, textvariable=status_var)
status_label.pack(side='right')

# image display
img_label = tk.Label(root, bd=2, relief='sunken', width=600, height=400)
img_label.pack(padx=8, pady=(4,8))

# Result label
result_label = tk.Label(root, text="No result", font=("Arial", 16, "bold"))
result_label.pack(pady=(4,8))

# Explanations text box (scrolled)
text_box = scrolledtext.ScrolledText(root, width=80, height=10, wrap=tk.WORD)
text_box.pack(padx=8, pady=(0,8))
text_box.configure(state='disabled')

root.mainloop()
