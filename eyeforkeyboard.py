# Overlay Data
transparent_color = "#000001"
font_size = 10
font = ('Impact', font_size)
label_bg = 'black'
label_color = "white"
from inference_sdk import InferenceHTTPClient
import tkinter as tk
import customtkinter as ctk
import cv2 as cv
import mediapipe as mp
import numpy as np
import time
from pygrabber.dshow_graph import FilterGraph

graph = FilterGraph()
ctk.set_appearance_mode('dark') # CTk Theme

# Setup Data
setup_window_name = "Setup"
cam_index = 0
width = 1280
height = 720
display_width = 800
max_points = 4
warp_ready = False
opencv_win = True
matrix = None
src_points = []
static_positions = []
cam_source = graph.get_input_devices()

# Keyboard Data
get_overlay = False
detected_filename = "detected_alpha.png"
keyboard_fileName = "keyboard_frame.png"
enable_preview = False
key_positions = []
key_labels = []
hand_positions = []
handedness = []
fingertips = [4, 8, 12, 16, 20]
def mouseCallback(event, x, y, flags, param): # Look for mouse click in cv
    global src_points, warp_ready

    if event == cv.EVENT_LBUTTONDOWN and len(src_points) < max_points:
        src_points.append([x, y])
        # print(f"Point {len(src_points)}: ({x}, {y})") # testing

        if len(src_points) == max_points:
            warp_ready = True
            # print("✅ All 4 points selected. Warping ready.") # testing
def keyDetector(): # use a frame to identify the keycaps locations
    global get_overlay, key_positions, static_positions, key_labels, width, height, w, h, matrix

    if warp_ready:
        cap = cv.VideoCapture(cam_index, cv.CAP_DSHOW) # Start capture with directShow
        cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        ret, frame = cap.read() 
        if not ret:
            cap.release()
            exit()
        h, w, _ = frame.shape
        cap.release() # stop the capture
        pts1 = np.array(src_points, dtype=np.float32)
        pts2 = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)

        matrix = cv.getPerspectiveTransform(pts1, pts2)
        frame = cv.warpPerspective(frame, matrix, (w, h))
        cv.imwrite(keyboard_fileName, frame) # saving the frame to a png

        # initilize the model
        CLIENT = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key="bEU3D1DGTYOEizYLiOKz"
        )
        key_result = CLIENT.infer(keyboard_fileName, model_id="keyboard-detection-v2/1")
        # print(key_result["predictions"]) # testing
        bgra_image = np.zeros((h, w, 4), dtype=np.uint8)
        key_labels = []
        key_positions = []
        for pred in key_result["predictions"]:
            # Break down json format
            x, y = pred["x"], pred["y"]
            wp, hp = pred["width"], pred["height"]
            label = pred["class"]

            # Get top-left & bottom-right
            x1 = int(x - wp / 2)
            y1 = int(y - hp / 2)
            x2 = int(x + wp / 2)
            y2 = int(y + hp / 2)
            # print(label, x, y) # testing
            cv.rectangle(bgra_image, (x1, y1), (x2, y2), (0, 255, 0, 128), 2)
            # Draw label
            cv.putText(bgra_image,
                    str(label),(x1, y1 + 10),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (255, 0, 255, 128),
                        1)
            if label=="keyboard":
                continue
            key_labels.append(label)
            key_positions.append(list((int(x), int(y))))
        cv.imwrite(detected_filename, bgra_image)
        static_positions = np.array(key_positions)
        key_labels = np.array(key_labels)
        get_overlay = True
def startHandTrack(): # Initialize MediaPipe Hands
    global hands, mp_drawing, mp_drawing_styles, mp_hands

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        max_num_hands = 2,
        model_complexity = 1,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5)
def setup(width, height): # Setting up the camera feed
    global src_points, warp_ready, opencv_win, setup_window_name, matrix
    cv.destroyAllWindows()
    cap = cv.VideoCapture(cam_index, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cv.namedWindow(setup_window_name)
    cv.setMouseCallback(setup_window_name, mouseCallback)

    # Get hight, width of frame
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture webcam frame")
    h, w, _ = frame.shape
    # print("height:",h," width:",w) # testing

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display_frame = frame.copy()

        # Draw selected points
        for pt in src_points:
            cv.circle(display_frame, tuple(pt), 5, (0, 255, 0), -1)
        # If 4 points are selected, apply warp
        if warp_ready:
            if opencv_win: # to close any previously opened cv windows
                opencv_win = False
                cv.destroyAllWindows()
            pts1 = np.array(src_points, dtype=np.float32)
            pts2 = np.array([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ], dtype=np.float32)

            matrix = cv.getPerspectiveTransform(pts1, pts2)
            warped = cv.warpPerspective(frame, matrix, (w, h))
            cv.imshow(f"{setup_window_name} (Completed)", warped)
        else:
            cv.imshow(setup_window_name, display_frame)
        key = cv.waitKey(1)
        
        if key == ord('r'):  # Reset points
            src_points = []
            warp_ready = False
            print("🔄 Reset points.")
            opencv_win = True
        elif key & 0xFF == 27:  # ESC to exit
            cap.release()
            cv.destroyAllWindows()
            break
    cap.release()
def getMinDistance(hand_result): # Gets the distance between fingertips and key centers
    global ft, key_labels, hand_positions, static_positions, h, w, key_caps, handedness
    hand_positions.clear()
    if hand_result.multi_hand_landmarks:
        # print(hand_result.multi_handedness) # testing
        for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
            handedness.append(hand_result.multi_handedness[idx].classification[0].label)
            # score = hand_result.multi_handedness[idx].classification[0].score # testing
            # print(f"Hand {idx+1}: {handedness} (confidence: {score:.2f})") # testing
            for ft in fingertips:
                lm = hand_landmarks.landmark[ft]
                hand_positions.append(list((int(lm.x*w), int(lm.y*h))))

        dynamic_positions = np.array(hand_positions)
        distances = np.linalg.norm(static_positions[:, None, :] - dynamic_positions[None, :, :], axis=2)
        # min_distances = np.min(distances, axis=0)  # Closest static point to each dynamic point|||addition 
        closest_static_indices = np.argmin(distances, axis=0)
        # print(min_distances,"::::",key_labels[closest_static_indices]) #testing
        return key_labels[closest_static_indices], handedness
    else:
        return [], []
def oneFrame(overlay_value): # Everything happening in a one frame when tracking
    global key_caps, handedness, cap, enable_preview

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture webcam frame")
    if warp_ready:
        frame = cv.warpPerspective(frame, matrix, (w, h))
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # Hand tracking
    frame.flags.writeable = False # To improve the proformance
    results = hands.process(frame)
    # print(results) # testing
    key_caps = None
    handedness.clear()
    if results.multi_hand_landmarks:
        key_caps, handedness = getMinDistance(results)
    # print(key_caps, ":::", handedness) #testing
    if enable_preview:
        frame.flags.writeable = True
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        if get_overlay:
            frame = (overlay_value * frame).astype(np.uint8)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        frame = cv.resize(frame, (640, 480), interpolation=cv.INTER_AREA)
        # print("Updating") # testing
        cv.imshow("Preview", frame) 
        if cv.waitKey(1) & 0xFF == 27:
            enable_preview = False
            exit()

    return key_caps, handedness

class SetupWindow(ctk.CTkToplevel):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.title(setup_window_name)
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        self.geometry(f'200x500+{screen_w-250}+{int(screen_h/2-250)}')
        self.resizable(False,False)
        self.focus()
        self.overrideredirect(True)
        # self.bind('<Escape>', lambda event: self.quit()) # quick
        self.rowconfigure((0,1,2,3,4,5,6,7,8,9,10), weight = 1, uniform = 'b')
        self.columnconfigure((0,1,2), weight = 1, uniform = 'b')
        
        # Widgets
        self.scale = ctk.CTkLabel(self, text = "Scale", corner_radius = 10)
        self.opasity = ctk.CTkLabel(self, text = "Opacity", corner_radius = 10)
        self.scale_slider = ctk.CTkSlider(self, from_ = 1, to = 5, variable = master.scale_var, command = master.setScale, border_width = 10)
        self.alpha_slider = ctk.CTkSlider(self, from_ = 0.1, to = 1, variable = master.alpha_var, command = master.setAlpha, border_width= 10)
        self.video_width = ctk.CTkEntry(self, placeholder_text = "Video Width in px", textvariable = master.camera_width)
        self.video_height = ctk.CTkEntry(self, placeholder_text = "Video Hieght in px", textvariable = master.camera_height)
        self.camera_input = ctk.CTkOptionMenu(self, values = cam_source, command = master.setSource, variable = master.source_str)
        self.apply_pref_button = ctk.CTkButton(self, text = "A", command = master.settingPreference, fg_color = "green")
        self.setup_button = ctk.CTkButton(self, textvariable = master.button_string_setup, command = self.startsetting)
        self.capture_button = ctk.CTkButton(self, textvariable = master.button_capture_string, command = master.capture, state = "disabled")
        self.stop_button = ctk.CTkButton(self, text = "Stop Tracking", command = master.stop, fg_color = "red")
        self.hide_button = ctk.CTkButton(self, text = "Hide", command = lambda : self.withdraw(), fg_color = "gray")
        self.preview_button = ctk.CTkButton(self, text = "Preview", command = master.openPreview)
        self.exit_button = ctk.CTkButton(self, text = "Exit", command = lambda : self.quit(), fg_color = "gray")
        self.key_binding = ctk.CTkLabel(self, text = "Open Settings Tray : SHIFT+ALT+S\nExit Video Window : Esc\nReset Point : r", fg_color = label_bg)
        # Layout
        self.scale.grid(row = 0, column = 0)
        self.opasity.grid(row = 1, column = 0)
        self.scale_slider.grid(row = 0, column = 1, sticky = "ew", columnspan = 3)
        self.alpha_slider.grid(row = 1, column = 1, sticky = "ew", columnspan = 3)
        self.camera_input.grid(row = 2, column = 0, sticky = "ews", columnspan = 3)
        self.video_width.grid(row = 3, column = 0, sticky = "news", columnspan = 3)
        self.video_height.grid(row = 4, column = 0, sticky = "news", columnspan = 3)
        self.apply_pref_button.grid(row = 5, column = 0, sticky = "news")
        self.setup_button.grid(row = 5, column = 1, sticky = "news", columnspan = 2)
        self.capture_button.grid(row = 6, column = 0, sticky = "news", columnspan = 3)
        self.stop_button.grid(row = 7, column = 0, sticky = "news", columnspan = 3)
        self.hide_button.grid(row = 8, column = 0, sticky = "news", columnspan = 3)
        self.preview_button.grid(row = 9, column = 0, sticky = "news", columnspan = 2)
        self.exit_button.grid(row = 9, column = 2, sticky = "news")
        self.key_binding.grid(row = 10, column = 0, sticky = "news", columnspan = 3)

        if warp_ready:
            self.capture_button.configure(state = "normal")
    def startsetting(self):
        self.master.settingUp()
        self.capture_button.configure(state = "normal")
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

        # variables
        self.camera_width = tk.IntVar(value = 1280)
        self.camera_height = tk.IntVar(value = 720)
        self.scale_var = tk.DoubleVar(value = 1.0)
        self.alpha_var = tk.DoubleVar(value = 1.0)
        self.source_str = tk.StringVar(value = cam_source[0])
        self.button_capture_string = tk.StringVar(value = "Start Tracking")
        self.button_string_setup = tk.StringVar(value = "Setup")
        # check for pref
        self.readPreference()
        # print(src_points, type(src_points), warp_ready, type(warp_ready)) # testing
        #w,h
        self.width = 400
        self.height = 70

        # Config window
        self.title("Keycaps Overlay")
        self.x_position, self.y_position = self.position()
        self.geometry(f'{int(self.width*self.scale_var.get())}x{int(self.height*self.scale_var.get())}+{self.x_position}+{self.y_position}')
        self.resizable(False,False)
        self.overrideredirect(True)
        self.attributes("-alpha", self.alpha_var.get())
        self.attributes("-transparentcolor", transparent_color)
        self.attributes("-topmost", True)
        self.update()

        # Bind
        # self.bind('<Escape>', lambda event: self.quit()) # quick
        self.bind('<Alt-Shift-KeyPress-S>', self.openSetupWindow)
        self.bind('<ButtonPress-1>', self.click)
        self.bind('<B1-Motion>', self.offset)


        # layout
        Overlay(self, fg_color = 'transparent')
        self.toplevel_window = None

        self.mainloop()
    
    def click(self, event):
        global click_x, click_y, screen_x, screen_y
        click_x = event.x_root
        click_y = event.y_root
        screen_x = self.winfo_x()
        screen_y = self.winfo_y()
    
    def offset(self, event):
        dx = event.x_root - click_x
        dy = event.y_root - click_y
        self.geometry(f"{screen_x+dx}+{screen_y+dy}")
    
    def position(self):
        x = int((self.winfo_screenwidth() - self.width)/2)
        y = self.winfo_screenheight() - 100
        return x, y
    
    def openSetupWindow(self, event):
        if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
            self.toplevel_window = SetupWindow(self)
        else:
            self.toplevel_window.focus()
            self.toplevel_window.deiconify()
    
    def readPreference(self):
        global cam_index, width, height, src_points, warp_ready
        try:
            with open("preference.txt", "r") as f:
                self.camera_width.set(f.readline().split()[1])
                width = self.camera_width.get()
                self.camera_height.set(f.readline().split()[1])
                height = self.camera_height.get()
                self.scale_var.set(f.readline().split()[1])
                self.alpha_var.set(f.readline().split()[1])
                cam_index = int(f.readline().split()[1])
                line = f.readline().split()
                if not line[1] == "[]":
                    for i in range(1,9,2):
                        src_points.append([int(line[i]), int(line[i+1])])
                if f.readline().split()[1] == "False":
                    warp_ready = False
                else:
                    warp_ready = True
        except:
            pass
        finally:
            self.source_str.set(cam_source[cam_index])

    def settingPreference(self):
        global cam_index
        with open("preference.txt", "w") as f:
            f.write(f"width {str(self.camera_width.get())}\n")
            f.write(f"height {str(self.camera_height.get())}\n")
            f.write(f"scale {str(self.scale_var.get())}\n")
            f.write(f"alpha {str(self.alpha_var.get())}\n")
            f.write(f"camera {cam_index}\n")
            f.write("points")
            for x, y in src_points:
                f.write(f" {x} {y}")
            f.write("\n")    
            f.write(f"wrap {str(warp_ready)}\n")

    def setScale(self, scale_value):
        global font_size
        self.geometry(f'{int(self.width*scale_value)}x{int(self.height*scale_value)}')

    def setAlpha(self, alpha_value):
        self.attributes("-alpha", alpha_value)

    def setSource(self, choice):
        global cam_index, cam_source
        for i in range(len(cam_source)):
            if cam_source[i] == choice:
                cam_index = i
    
    def settingUp(self):
        time.sleep(1.00)
        setup(self.camera_width.get(), self.camera_height.get())
    
    def capture(self):
        global overlay_value, cap
        keyDetector()
        startHandTrack()
        self.button_capture_string.set("Tracking...")

        if get_overlay:
            overlay = cv.imread(detected_filename, cv.IMREAD_UNCHANGED)
            overlay_bgr = overlay[:, :, :3]
            overlay_alpha = overlay[:, :, 3:] / 255.0 
            overlay_value = overlay_alpha * overlay_bgr + (1 - overlay_alpha)
        else:
            overlay_value = None
        cap = cv.VideoCapture(cam_index, cv.CAP_DSHOW)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, self.camera_width.get())
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.camera_height.get())
        self.toplevel_window.capture_button.configure(state = "disable")
        self.running()

    def running(self):
        global overlay_value, after_id

        key_caps, handedness = oneFrame(overlay_value)
        if len(handedness) == 2:
            self.l1.set(key_caps[0])
            self.l2.set(key_caps[1])
            self.l3.set(key_caps[2])
            self.l4.set(key_caps[3])
            self.l5.set(key_caps[4])
            self.r1.set(key_caps[5])
            self.r2.set(key_caps[6])
            self.r3.set(key_caps[7])
            self.r4.set(key_caps[8])
            self.r5.set(key_caps[9])
        elif len(handedness) == 1:
            if handedness[0] == 'left':
                self.l1.set(key_caps[0])
                self.l2.set(key_caps[1])
                self.l3.set(key_caps[2])
                self.l4.set(key_caps[3])
                self.l5.set(key_caps[4])
            else:
                self.r1.set(key_caps[0])
                self.r2.set(key_caps[1])
                self.r3.set(key_caps[2])
                self.r4.set(key_caps[3])
                self.r5.set(key_caps[4])                

        after_id = self.after(1, self.running)
  
    def stop(self):
        global cap

        cap.release()
        cv.destroyAllWindows()
        self.after_cancel(after_id)
        self.button_capture_string.set("Start Tracking")
        self.toplevel_window.capture_button.configure(state = "normal")

    def openPreview(self):
        global enable_preview
        enable_preview = True
App(fg_color = transparent_color)