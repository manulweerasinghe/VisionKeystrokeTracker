import tkinter as tk
import customtkinter as ctk
import cv2 as cv
import time
from settings import *
from main import *


ctk.set_appearance_mode('dark')

class App(ctk.CTk):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Finger data
        self.l1 = tk.StringVar(value = "L1")
        self.l2 = tk.StringVar(value = "L2")
        self.l3 = tk.StringVar(value = "L3")
        self.l4 = tk.StringVar(value = "L4")
        self.l5 = tk.StringVar(value = "L5")
        
        self.r1 = tk.StringVar(value = "R1")
        self.r2 = tk.StringVar(value = "R2")
        self.r3 = tk.StringVar(value = "R3")
        self.r4 = tk.StringVar(value = "R4")
        self.r5 = tk.StringVar(value = "R5")

        self.width = 400
        self.height = 70

        # Config window
        self.title("System Tray App")
        self.x_position, self.y_position = self.position()
        self.geometry(f'{self.width}x{self.height}+{self.x_position}+{self.y_position}')
        self.resizable(False,False)
        self.overrideredirect(True)
        self.attributes("-alpha", 0.5)
        self.attributes("-topmost", True)
        self.update()

        # Bind
        self.bind('<Escape>', lambda event: self.quit())
        self.bind('<Alt-Shift-KeyPress-S>', setup)
        self.bind('<Alt-Shift-KeyPress-K>', self.detector)
        self.bind('<Alt-Shift-KeyPress-T>', self.tracking)
        self.bind('<Alt-Shift-KeyPress-V>', self.preview)

        # layout
        Results(self, fg_color = 'transparent')

        self.mainloop()

    def position(self):
        x = int((self.winfo_screenwidth() - self.width)/2)
        y = self.winfo_screenheight() - 100
        return x, y
    
    def detector(self, event):
        global enable_handtracking, cam_index
        time.sleep(5.00)
        keyboard_result, height, width = keyDetector(2)
        seperator(keyboard_result, height, width)
        startHandTrack()

    def tracking(self, event):
        global enable_annote, enable_handtracking

        if enable_preview:
            oneFrame(self.cap, self.w, self.h) 
            self.after(0, self.tracking)

    def preview(self, event):
        global enable_preview, enable_annote
        # set variables
        enable_annote = True
        enable_preview = False
        show()

class Results(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.rowconfigure((0,1,2), weight = 1, uniform = 'b')
        self.columnconfigure((0,1,2,3,4,5,6,7,8,9), weight = 1, uniform = 'b')
        # Left Labels
        finger_l1 = ctk.CTkLabel(self, textvariable = master.l1, font = font, text_color = label_color, fg_color = label_bg, corner_radius = 5)
        finger_l2 = ctk.CTkLabel(self, textvariable = master.l2, font = font, text_color = label_color, fg_color = label_bg, corner_radius = 5)
        finger_l3 = ctk.CTkLabel(self, textvariable = master.l3, font = font, text_color = label_color, fg_color = label_bg, corner_radius = 5)
        finger_l4 = ctk.CTkLabel(self, textvariable = master.l4, font = font, text_color = label_color, fg_color = label_bg, corner_radius = 5)
        finger_l5 = ctk.CTkLabel(self, textvariable = master.l5, font = font, text_color = label_color, fg_color = label_bg, corner_radius = 5)
        # Right Labels
        finger_r1 = ctk.CTkLabel(self, textvariable = master.r1, font = font, text_color = label_color, fg_color = label_bg, corner_radius = 5)
        finger_r2 = ctk.CTkLabel(self, textvariable = master.r2, font = font, text_color = label_color, fg_color = label_bg, corner_radius = 5)
        finger_r3 = ctk.CTkLabel(self, textvariable = master.r3, font = font, text_color = label_color, fg_color = label_bg, corner_radius = 5)
        finger_r4 = ctk.CTkLabel(self, textvariable = master.r4, font = font, text_color = label_color, fg_color = label_bg, corner_radius = 5)
        finger_r5 = ctk.CTkLabel(self, textvariable = master.r5, font = font, text_color = label_color, fg_color = label_bg, corner_radius = 5)

        # Layout
        self.pack(side = "left", fill = "both", expand = True)
        finger_l1.grid(row = 2, column = 0, sticky = "news")
        finger_l2.grid(row = 1, column = 1, sticky = "news")
        finger_l3.grid(row = 0, column = 2, sticky = "news")
        finger_l4.grid(row = 1, column = 3, sticky = "news")
        finger_l5.grid(row = 2, column = 4, sticky = "news")

        finger_r1.grid(row = 2, column = 5, sticky = "news")
        finger_r2.grid(row = 1, column = 6, sticky = "news")
        finger_r3.grid(row = 0, column = 7, sticky = "news")
        finger_r4.grid(row = 1, column = 8, sticky = "news")
        finger_r5.grid(row = 2, column = 9, sticky = "news")

if __name__ == "__main__":
    App()

