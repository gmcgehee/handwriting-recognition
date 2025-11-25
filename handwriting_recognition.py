import customtkinter as ctk
import numpy as np
import joblib
from PIL import Image
from matplotlib import pyplot as plt

class Window(ctk.CTk):

    def __init__(self, model):
        super().__init__()
        self.geometry('500x500')
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.img_px = np.zeros((1, 28, 28), dtype=np.uint8)
        self.model = knn = joblib.load('knn_hw.pkl')

        # GPT 3/4/25, edited by Geist McGehee
        self.CANVAS_SIZE = 300
        self.canvas = ctk.CTkCanvas(self, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE, bg='white')
        #self.canvas.pack(pady=10)
        self.canvas.place(relx='0.5', rely='0.3', anchor='center')
        self.canvas.bind("<B1-Motion>", self.draw)

        self.predict_button = ctk.CTkButton(self, text="Predict", command=self.predict)
        self.predict_button.place(relx='0.5', rely='0.8', anchor='center')
        # End GPT
        
        self.reset_button = ctk.CTkButton(self, text='Reset Canvas', command=self.reset)
        self.reset_button.place(relx='0.5', rely='0.9', anchor='center')

    # GPT 3/4/25
    def draw(self, event): 
        oval_size = 5
        x, y = event.x, event.y
        grid_x = min(27, max(0, int(x / self.CANVAS_SIZE * 28))) # GPT 3/7, black magic code that just works
        grid_y = min(27, max(0, int(y / self.CANVAS_SIZE * 28))) # GPT 3/7
        self.img_px[0, grid_y, grid_x] = np.uint8(255)
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="black", outline='black')

    def reset(self):
        self.canvas.delete('all')
        self.img_px = np.zeros((1, 28, 28), dtype=np.uint8)
        #self.canvas = ctk.CTkCanvas(self, width=300, height=300, bg='white')
        #self.canvas.place(relx='0.5', rely='0.3', anchor='center')
        #self.canvas.bind("<B1-Motion>", self.draw)
       
    def predict(self):
        digit = self.img_px[0].flatten().reshape(1, -1)
        if not digit.any():
            return
        prediction = self.model.predict(digit)
        self.label = ctk.CTkLabel(self, text=f'Prediction: {prediction[0]}')
        self.label.place(relx='0.5', rely='0.7', anchor='center')
        # Convert to Pillow image and save
        #pimage = Image.fromarray(digit.reshape(28, 28))
        #pimage.show()



if __name__ == '__main__':
    knn = joblib.load('knn_hw.pkl')
    window = Window(model=knn)
    window.mainloop()