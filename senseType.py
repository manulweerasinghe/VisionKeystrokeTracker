import tkinter as tk
import customtkinter as ctk
import cv2 as cv
from PIL import Image, ImageTk
ctk.set_appearance_mode('dark') # CTk Theme
# Setup Data
setup_window_name = "Setup Window"
cam_index = 3
width = 1280
height = 720
display_width = 800

# Overlay Data
font = ('Impact', 20)
label_bg = 'black'
label_color = "white"
def errorMessage(self):
    self.display_string.set("Failed to capture frame")
    self.button_string.set("Start Capture")
    self.cap.release()
    raise RuntimeError("Failed to capture frame")
# Select a video input and setup
class SetupWindow(ctk.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title(setup_window_name)
        self.geometry(f'{display_width + 100}x500')
        self.bind('<Escape>', lambda event: self.quit())
        self.protocol('WM_DELETE_WINDOW', self.destructor)
        self.display_string = tk.StringVar(value = "NO video feed")
        self.button_string = tk.StringVar(value = "Start capture")
        self.start_capture = False
        # Widgets
        self.display_frame = ctk.CTkLabel(self, textvariable = self.display_string)
        self.button_menu = ctk.CTkFrame(self)
        self.capture_button = ctk.CTkButton(self.button_menu, textvariable = self.button_string, command = self.capture)
        self.stop_button = ctk.CTkButton(self.button_menu, text = "Stop Capture", command = self.stop)
        # Layout
        self.display_frame.pack(side = "left", expand = True, fill = "both")
        self.button_menu.pack(side = "left", expand = True, fill = "both")
        self.capture_button.pack(side = "top", expand = True, fill = "both")
        self.stop_button.pack(side = "top", expand = True, fill = "both")
        # Capture
        self.cap = cv.VideoCapture(cam_index, cv.CAP_DSHOW)
        ret, frame = self.cap.read()
        if not ret:
            errorMessage(self)
        self.f_height, self.f_width, _ = frame.shape
        self.img_ratio = self.f_width / self.f_height
    
    def capture(self):
        self.start_capture = True
        self.button_string.set("Capturing")  
        self.display_string.set("")
        self.scalling()
        self.oneFrame()
    
    def stop(self):
        self.start_capture = False
        
    def oneFrame(self):
        global start_capture

        ret, frame = self.cap.read()
        if ret and self.start_capture:
            # frame = cv.resize(frame, (self.new_width,self.new_height), interpolation = cv.ANT)    
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
            frame_pil = Image.fromarray(frame)
            resized = frame_pil.resize((self.new_width, self.new_height), Image.Resampling.LANCZOS)
            frame_ctk = ctk.CTkImage(light_image = resized, dark_image = resized, size = (self.new_width, self.new_height))
            # frame_tk = ImageTk.PhotoImage(image = frame_pil)
            # self.display_frame.frame_tk = frame_tk
            self.display_frame.configure(image = frame_ctk)
            self.display_frame.after(1, self.oneFrame)
        elif self.start_capture:
            errorMessage(self)

    def scalling(self):
        global display_width
        
        window_width = self.winfo_width
        self.new_width = int(display_width)
        self.new_height = int(self.new_width * self.img_ratio)
        
    def destructor(self):
        self.destroy()
        self.cap.release()




# Overlay with keycaps
class Overlay(ctk.CTkFrame):
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
class App(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        self.bind('<Alt-Shift-KeyPress-S>', self.openSetupWindow)

        self.toplevel_window = None
        # layout
        Overlay(self, fg_color = 'transparent')

        self.mainloop()

    def openSetupWindow(self, event):
        if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
            self.toplevel_window = SetupWindow(self)
        else:
            self.toplevel_window.focus()

    def position(self):
        x = int((self.winfo_screenwidth() - self.width)/2)
        y = self.winfo_screenheight() - 100
        return x, y


App()