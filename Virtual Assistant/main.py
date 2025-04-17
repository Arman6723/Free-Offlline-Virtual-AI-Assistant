import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import json
import os  # Added for os.makedirs, os.path, etc.
import shutil
from PIL import Image, ImageTk
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPProcessor, CLIPModel
import torch
import PyPDF2
import threading
import time
from diffusers import StableDiffusionPipeline
import cv2  # Added for OpenCV functionality
import numpy as np
import pygame
import random
from pathlib import Path  # Added for Path in VideoApp

try:
    import tkinterdnd2 as tkdnd
    TKDND_AVAILABLE = True
except ImportError:
    TKDND_AVAILABLE = False
    tk.messagebox.showwarning("Warning", "tkinterdnd2 not found. Drag-and-drop disabled.")

ATTACHMENTS_DIR = "attachments"
GENERATED_IMAGES_DIR = "generated_images"
CHAT_HISTORY_FILE = "chat_history.json"
CONFIG_FILE = "config.json"
VIDEO_TEMP_DIR = "C:/VideoAppTempFiles"

os.makedirs(ATTACHMENTS_DIR, exist_ok=True)
os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)
os.makedirs(VIDEO_TEMP_DIR, exist_ok=True)

def initialize_chat_history():
    default_history = {"chats": {}}
    if not os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'w') as f:
            json.dump(default_history, f)
    else:
        try:
            with open(CHAT_HISTORY_FILE, 'r') as f:
                data = json.load(f)
                if not isinstance(data, dict) or "chats" not in data:
                    with open(CHAT_HISTORY_FILE, 'w') as f:
                        json.dump(default_history, f)
        except json.JSONDecodeError:
            with open(CHAT_HISTORY_FILE, 'w') as f:
                json.dump(default_history, f)

initialize_chat_history()

class SettingsGUI:
    def __init__(self, root, callback):
        self.root = root
        self.root.title("OmniCore Settings")
        self.root.geometry("400x500")
        self.callback = callback
        self.config = self.load_config()
        self.create_gui()

    def load_config(self):
        default_config = {
            "performance_mode": "High",
            "image_quality": "High",
            "max_chat_length": "Long",
            "power_level": "Balanced"  # Ensure power_level is included
        }
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    # Ensure all keys exist, fill missing ones from default
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except (json.JSONDecodeError, IOError):
                # If file is corrupted or unreadable, return default
                return default_config
        return default_config

    def save_config(self):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=4)

    def create_gui(self):
        tk.Label(self.root, text="Power Level:").pack(pady=5)
        self.power_var = tk.StringVar(value=self.config["power_level"])
        frame = tk.Frame(self.root)
        frame.pack()
        tk.Radiobutton(frame, text="Eco", variable=self.power_var, value="Eco").pack(side=tk.LEFT)
        tk.Radiobutton(frame, text="Balanced", variable=self.power_var, value="Balanced").pack(side=tk.LEFT)
        tk.Radiobutton(frame, text="Max", variable=self.power_var, value="Max").pack(side=tk.LEFT)
        tk.Button(frame, text="?", command=lambda: tk.messagebox.showinfo("Power Level", "Eco: Minimal resources, fastest. Balanced: Good performance and quality. Max: Best quality, high resource use.")).pack(side=tk.LEFT, padx=5)

        tk.Label(self.root, text="Performance Mode:").pack(pady=5)
        self.performance_var = tk.StringVar(value=self.config["performance_mode"])
        frame = tk.Frame(self.root)
        frame.pack()
        tk.Radiobutton(frame, text="Low PC Mode", variable=self.performance_var, value="Low").pack(side=tk.LEFT)
        tk.Radiobutton(frame, text="High Performance", variable=self.performance_var, value="High").pack(side=tk.LEFT)
        tk.Button(frame, text="?", command=lambda: tk.messagebox.showinfo("Performance Mode", "Low PC Mode reduces resource usage. High Performance uses more resources for better quality.")).pack(side=tk.LEFT, padx=5)

        tk.Label(self.root, text="Image Generation Quality:").pack(pady=5)
        self.image_quality_var = tk.StringVar(value=self.config["image_quality"])
        frame = tk.Frame(self.root)
        frame.pack()
        tk.Radiobutton(frame, text="Low (faster)", variable=self.image_quality_var, value="Low").pack(side=tk.LEFT)
        tk.Radiobutton(frame, text="High (better quality)", variable=self.image_quality_var, value="High").pack(side=tk.LEFT)
        tk.Button(frame, text="?", command=lambda: tk.messagebox.showinfo("Image Quality", "Low: ~15s. High: ~30s, better quality.")).pack(side=tk.LEFT, padx=5)

        tk.Label(self.root, text="Default Chat Length:").pack(pady=5)
        self.chat_length_var = tk.StringVar(value=self.config["max_chat_length"])
        frame = tk.Frame(self.root)
        frame.pack()
        tk.Radiobutton(frame, text="Short", variable=self.chat_length_var, value="Short").pack(side=tk.LEFT)
        tk.Radiobutton(frame, text="Long", variable=self.chat_length_var, value="Long").pack(side=tk.LEFT)
        tk.Button(frame, text="?", command=lambda: tk.messagebox.showinfo("Max Chat Length", "Short: ~150 words, faster. Long: ~300 words, detailed.")).pack(side=tk.LEFT, padx=5)

        tk.Button(self.root, text="Start OmniCore", command=self.start).pack(pady=20)

    def start(self):
        self.config["performance_mode"] = self.performance_var.get()
        self.config["image_quality"] = self.image_quality_var.get()
        self.config["max_chat_length"] = self.chat_length_var.get()
        self.config["power_level"] = self.power_var.get()
        self.save_config()
        self.root.destroy()
        self.callback(self.config)

class SnakeGame:
    def __init__(self, canvas, on_exit):
        self.canvas = canvas
        self.on_exit = on_exit
        self.width = 400
        self.height = 400
        self.cell_size = 20
        self.snake = [(5, 5)]
        self.direction = (1, 0)
        self.food = self.spawn_food()
        self.score = 0
        self.game_over = False
        self.bind_controls()
        self.draw()

    def bind_controls(self):
        self.canvas.bind("<w>", lambda e: self.set_direction((0, -1)))
        self.canvas.bind("<a>", lambda e: self.set_direction((-1, 0)))
        self.canvas.bind("<s>", lambda e: self.set_direction((0, 1)))
        self.canvas.bind("<d>", lambda e: self.set_direction((1, 0)))
        self.canvas.bind("<Up>", lambda e: self.set_direction((0, -1)))
        self.canvas.bind("<Left>", lambda e: self.set_direction((-1, 0)))
        self.canvas.bind("<Down>", lambda e: self.set_direction((0, 1)))
        self.canvas.bind("<Right>", lambda e: self.set_direction((1, 0)))
        self.canvas.bind("<q>", lambda e: self.restart())
        self.canvas.bind("<e>", lambda e: self.restart())
        self.canvas.bind("<Button-1>", lambda e: self.restart())
        self.canvas.bind("<Button-3>", lambda e: self.restart())
        self.canvas.bind("<Escape>", lambda e: self.exit_game())

    def set_direction(self, new_dir):
        if (new_dir[0] != -self.direction[0] or new_dir[1] != -self.direction[1]) and not self.game_over:
            self.direction = new_dir

    def spawn_food(self):
        while True:
            x = random.randint(0, self.width // self.cell_size - 1)
            y = random.randint(0, self.height // self.cell_size - 1)
            if (x, y) not in self.snake:
                return (x, y)

    def update(self):
        if self.game_over:
            return
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)
        
        if (new_head[0] < 0 or new_head[0] >= self.width // self.cell_size or
            new_head[1] < 0 or new_head[1] >= self.height // self.cell_size or
            new_head in self.snake):
            self.game_over = True
            return
        
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            self.food = self.spawn_food()
        else:
            self.snake.pop()
        
        self.draw()
        self.canvas.after(100, self.update)

    def draw(self):
        self.canvas.delete("all")
        for x, y in self.snake:
            self.canvas.create_rectangle(
                x * self.cell_size, y * self.cell_size,
                (x + 1) * self.cell_size, (y + 1) * self.cell_size,
                fill="green"
            )
        fx, fy = self.food
        self.canvas.create_oval(
            fx * self.cell_size, fy * self.cell_size,
            (fx + 1) * self.cell_size, (fy + 1) * self.cell_size,
            fill="red"
        )
        self.canvas.create_text(
            10, 10, anchor="nw", text=f"Score: {self.score}", fill="white"
        )
        if self.game_over:
            self.canvas.create_text(
                self.width // 2, self.height // 2,
                text=f"Game Over! Score: {self.score}\nQ/E/Click to Restart",
                fill="white", justify="center"
            )

    def restart(self):
        self.snake = [(5, 5)]
        self.direction = (1, 0)
        self.food = self.spawn_food()
        self.score = 0
        self.game_over = False
        self.draw()

    def exit_game(self):
        self.canvas.delete("all")
        self.canvas.unbind("<w>")
        self.canvas.unbind("<a>")
        self.canvas.unbind("<s>")
        self.canvas.unbind("<d>")
        self.canvas.unbind("<Up>")
        self.canvas.unbind("<Left>")
        self.canvas.unbind("<Down>")
        self.canvas.unbind("<Right>")
        self.canvas.unbind("<q>")
        self.canvas.unbind("<e>")
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<Button-3>")
        self.canvas.unbind("<Escape>")
        self.on_exit()

class VideoApp:
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.temp_folder = Path(VIDEO_TEMP_DIR)
        self.cap = None
        self.duration = 0
        self.videos = []
        self.codec = 'mp4v'
        self.scale_factor = 1.0
        self.custom_fps = None
        self.setup_gui()

    def setup_gui(self):
        self.notebook = ttk.Notebook(self.parent_frame)
        self.notebook.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        self.editor_tab = ttk.Frame(self.notebook)
        self.merger_tab = ttk.Frame(self.notebook)
        self.splitter_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.editor_tab, text="Editor")
        self.notebook.add(self.merger_tab, text="Merger")
        self.notebook.add(self.splitter_tab, text="Splitter")
        self.notebook.add(self.settings_tab, text="Settings")
        
        self.setup_editor_tab()
        self.setup_merger_tab()
        self.setup_splitter_tab()
        self.setup_settings_tab()

    def setup_editor_tab(self):
        tk.Label(self.editor_tab, text="Input Video:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.editor_input_entry = tk.Entry(self.editor_tab, width=50)
        self.editor_input_entry.grid(row=0, column=1, padx=10, pady=10)
        tk.Button(self.editor_tab, text="Browse", command=self.editor_load_video).grid(row=0, column=2, padx=10)
        
        tk.Label(self.editor_tab, text="Output Folder:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.editor_output_entry = tk.Entry(self.editor_tab, width=50)
        self.editor_output_entry.grid(row=1, column=1, padx=10, pady=10)
        tk.Button(self.editor_tab, text="Browse", command=lambda: self.browse_output(self.editor_output_entry)).grid(row=1, column=2, padx=10)
        
        tk.Label(self.editor_tab, text="Trim (start-end, sec):").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.editor_trim_start = tk.Entry(self.editor_tab, width=10)
        self.editor_trim_start.insert(0, "0")
        self.editor_trim_start.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        self.editor_trim_end = tk.Entry(self.editor_tab, width=10)
        self.editor_trim_end.insert(0, "10")
        self.editor_trim_end.grid(row=2, column=1, padx=70, pady=10, sticky="w")
        
        tk.Label(self.editor_tab, text="Crop (x,y,w,h):").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.editor_crop_x = tk.Entry(self.editor_tab, width=7)
        self.editor_crop_x.insert(0, "0")
        self.editor_crop_x.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        self.editor_crop_y = tk.Entry(self.editor_tab, width=7)
        self.editor_crop_y.insert(0, "0")
        self.editor_crop_y.grid(row=3, column=1, padx=60, pady=10, sticky="w")
        self.editor_crop_w = tk.Entry(self.editor_tab, width=7)
        self.editor_crop_w.insert(0, "100")
        self.editor_crop_w.grid(row=3, column=1, padx=110, pady=10, sticky="w")
        self.editor_crop_h = tk.Entry(self.editor_tab, width=7)
        self.editor_crop_h.insert(0, "100")
        self.editor_crop_h.grid(row=3, column=1, padx=160, pady=10, sticky="w")
        
        tk.Label(self.editor_tab, text="Brightness (-255 to 255):").grid(row=4, column=0, padx=10, pady=10, sticky="w")
        self.editor_brightness = tk.Entry(self.editor_tab, width=10)
        self.editor_brightness.insert(0, "0")
        self.editor_brightness.grid(row=4, column=1, padx=10, pady=10, sticky="w")
        
        tk.Label(self.editor_tab, text="Contrast (0-2):").grid(row=5, column=0, padx=10, pady=10, sticky="w")
        self.editor_contrast = tk.Entry(self.editor_tab, width=10)
        self.editor_contrast.insert(0, "1")
        self.editor_contrast.grid(row=5, column=1, padx=10, pady=10, sticky="w")
        
        tk.Button(self.editor_tab, text="Edit & Save", command=self.editor_save_video, bg="green", fg="white").grid(row=6, column=1, pady=10)
        tk.Button(self.editor_tab, text="Clear Temp Files", command=self.delete_temp_files, bg="red", fg="white").grid(row=7, column=1, pady=10)

    def setup_merger_tab(self):
        file_frame = ttk.Frame(self.merger_tab)
        file_frame.pack(pady=10, padx=10, fill=tk.X)
        
        ttk.Button(file_frame, text="Browse Files", command=self.merger_browse_files).pack(side=tk.LEFT, padx=5)
        self.merger_path_entry = ttk.Entry(file_frame)
        self.merger_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(file_frame, text="Add Path", command=self.merger_add_path).pack(side=tk.LEFT, padx=5)
        
        list_frame = ttk.Frame(self.merger_tab)
        list_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        self.merger_video_list = tk.Listbox(list_frame, selectmode=tk.SINGLE, height=10)
        self.merger_video_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.merger_video_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.merger_video_list.config(yscrollcommand=scrollbar.set)
        
        order_frame = ttk.Frame(self.merger_tab)
        order_frame.pack(pady=5)
        ttk.Button(order_frame, text="Move Up", command=self.merger_move_up).pack(side=tk.LEFT, padx=5)
        ttk.Button(order_frame, text="Move Down", command=self.merger_move_down).pack(side=tk.LEFT, padx=5)
        ttk.Button(order_frame, text="Remove", command=self.merger_remove_video).pack(side=tk.LEFT, padx=5)
        
        progress_frame = ttk.Frame(self.merger_tab)
        progress_frame.pack(pady=5, fill=tk.X, padx=10)
        self.merger_progress = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.merger_progress.pack()
        
        button_frame = ttk.Frame(self.merger_tab)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Merge Videos", command=self.merger_merge_videos, style="Green.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Delete Temp Files", command=self.delete_temp_files, style="Red.TButton").pack(side=tk.LEFT, padx=5)
        
        style = ttk.Style()
        style.configure("Green.TButton", background="green", foreground="white")
        style.configure("Red.TButton", background="red", foreground="white")
        
        if TKDND_AVAILABLE:
            self.merger_video_list.drop_target_register(tkdnd.DND_FILES)
            self.merger_video_list.dnd_bind('<<Drop>>', self.merger_handle_drop)

    def setup_splitter_tab(self):
        tk.Label(self.splitter_tab, text="Input Video:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.splitter_input_entry = tk.Entry(self.splitter_tab, width=50)
        self.splitter_input_entry.grid(row=0, column=1, padx=10, pady=10)
        tk.Button(self.splitter_tab, text="Browse", command=self.splitter_load_video).grid(row=0, column=2, padx=10)
        
        tk.Label(self.splitter_tab, text="Output Folder:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.splitter_output_entry = tk.Entry(self.splitter_tab, width=50)
        self.splitter_output_entry.grid(row=1, column=1, padx=10, pady=10)
        tk.Button(self.splitter_tab, text="Browse", command=lambda: self.browse_output(self.splitter_output_entry)).grid(row=1, column=2, padx=10)
        
        tk.Label(self.splitter_tab, text="Split Mode:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.split_mode = tk.StringVar(value="parts")
        tk.Radiobutton(self.splitter_tab, text="By Parts", variable=self.split_mode, value="parts").grid(row=2, column=1, padx=10, pady=5, sticky="w")
        tk.Radiobutton(self.splitter_tab, text="By Time (sec)", variable=self.split_mode, value="time").grid(row=2, column=1, padx=100, pady=5, sticky="w")
        
        tk.Label(self.splitter_tab, text="Number of Parts/Time (sec):").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.split_value = tk.Entry(self.splitter_tab, width=10)
        self.split_value.insert(0, "2")
        self.split_value.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        
        tk.Button(self.splitter_tab, text="Split Video", command=self.splitter_split_video, bg="green", fg="white").grid(row=4, column=1, pady=10)
        tk.Button(self.splitter_tab, text="Clear Temp Files", command=self.delete_temp_files, bg="red", fg="white").grid(row=5, column=1, pady=10)

    def setup_settings_tab(self):
        tk.Label(self.settings_tab, text="Codec Selection:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.codec_var = tk.StringVar(value="mp4v")
        tk.Radiobutton(self.settings_tab, text="MP4V", variable=self.codec_var, value="mp4v", command=self.update_codec).grid(row=0, column=1, padx=10, pady=5, sticky="w")
        tk.Radiobutton(self.settings_tab, text="XVID", variable=self.codec_var, value="XVID", command=self.update_codec).grid(row=0, column=1, padx=100, pady=5, sticky="w")
        
        tk.Label(self.settings_tab, text="Resolution Scale (0.1-2.0):").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.scale_entry = tk.Entry(self.settings_tab, width=10)
        self.scale_entry.insert(0, "1.0")
        self.scale_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        tk.Button(self.settings_tab, text="Apply Scale", command=self.update_scale).grid(row=1, column=2, padx=10)
        
        tk.Label(self.settings_tab, text="Custom FPS (leave blank for default):").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.fps_entry = tk.Entry(self.settings_tab, width=10)
        self.fps_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        tk.Button(self.settings_tab, text="Apply FPS", command=self.update_fps).grid(row=2, column=2, padx=10)

    def editor_load_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mkv *.avi")])
        if not file_path:
            return
        try:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                tk.messagebox.showerror("Error", "Can't open video.")
                self.cap = None
                return
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                tk.messagebox.showerror("Error", "Can't get video FPS.")
                self.cap.release()
                self.cap = None
                return
            frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.duration = frame_count / fps
            self.editor_input_entry.delete(0, tk.END)
            self.editor_input_entry.insert(0, file_path)
            self.editor_trim_end.delete(0, tk.END)
            self.editor_trim_end.insert(0, str(int(self.duration)))
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to load video: {str(e)}")
            if self.cap:
                self.cap.release()
            self.cap = None

    def editor_save_video(self):
        if not self.cap:
            tk.messagebox.showerror("Error", "Load a video first.")
            return
        output_path = self.editor_output_entry.get()
        if not output_path:
            tk.messagebox.showerror("Error", "Pick an output folder.")
            return
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path)
            except:
                tk.messagebox.showerror("Error", "Can't make folder.")
                return
        try:
            start = float(self.editor_trim_start.get())
            end = float(self.editor_trim_end.get())
            x, y = int(self.editor_crop_x.get()), int(self.editor_crop_y.get())
            w, h = int(self.editor_crop_w.get()), int(self.editor_crop_h.get())
            brightness = float(self.editor_brightness.get())
            contrast = float(self.editor_contrast.get())
            if start < 0 or end > self.duration or start >= end:
                tk.messagebox.showerror("Error", "Invalid trim times.")
                return
            frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > frame_width or y+h > frame_height:
                tk.messagebox.showerror("Error", "Invalid crop values.")
                return
            
            fps = self.custom_fps if self.custom_fps else self.cap.get(cv2.CAP_PROP_FPS)
            start_frame = int(start * fps)
            end_frame = int(end * fps)
            scaled_w, scaled_h = int(w * self.scale_factor), int(h * self.scale_factor)
            output_file = os.path.join(output_path, "edited_video.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            out = cv2.VideoWriter(output_file, fourcc, fps, (scaled_w, scaled_h))
            if not out.isOpened():
                raise Exception("Can't initialize video writer.")
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for i in range(start_frame, end_frame):
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame[y:y+h, x:x+w]
                frame = frame * contrast + brightness
                frame = np.clip(frame, 0, 255).astype(np.uint8)
                if self.scale_factor != 1.0:
                    frame = cv2.resize(frame, (scaled_w, scaled_h))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
            
            out.release()
            tk.messagebox.showinfo("Success", "Video edited and saved.")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Save failed: {str(e)}")

    def merger_handle_drop(self, event):
        if not TKDND_AVAILABLE:
            return
        files = self.parent_frame.splitlist(event.data)
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.merger_add_video(file)

    def merger_browse_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        for file in files:
            self.merger_add_video(file)

    def merger_add_path(self):
        path = self.merger_path_entry.get()
        if os.path.isfile(path) and path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            self.merger_add_video(path)
            self.merger_path_entry.delete(0, tk.END)
        else:
            tk.messagebox.showerror("Error", "Invalid video file.")

    def merger_add_video(self, path):
        if path not in self.videos:
            self.videos.append(path)
            self.merger_video_list.insert(tk.END, os.path.basename(path))
            temp_path = self.temp_folder / os.path.basename(path)
            shutil.copy(path, temp_path)

    def merger_remove_video(self):
        selection = self.merger_video_list.curselection()
        if selection:
            index = selection[0]
            video_path = self.videos[index]
            temp_path = self.temp_folder / os.path.basename(video_path)
            if temp_path.exists():
                temp_path.unlink()
            self.videos.pop(index)
            self.merger_video_list.delete(index)

    def merger_move_up(self):
        selection = self.merger_video_list.curselection()
        if selection and selection[0] > 0:
            index = selection[0]
            self.videos[index], self.videos[index-1] = self.videos[index-1], self.videos[index]
            self.merger_video_list.delete(index)
            self.merger_video_list.insert(index-1, os.path.basename(self.videos[index-1]))
            self.merger_video_list.selection_clear(0, tk.END)
            self.merger_video_list.selection_set(index-1)

    def merger_move_down(self):
        selection = self.merger_video_list.curselection()
        if selection and selection[0] < len(self.videos) - 1:
            index = selection[0]
            self.videos[index], self.videos[index+1] = self.videos[index+1], self.videos[index]
            self.merger_video_list.delete(index)
            self.merger_video_list.insert(index+1, os.path.basename(self.videos[index+1]))
            self.merger_video_list.selection_clear(0, tk.END)
            self.merger_video_list.selection_set(index+1)

    def merger_merge_videos(self):
        if len(self.videos) < 1:
            tk.messagebox.showerror("Error", "Add at least one video.")
            return
        
        output_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
        if not output_path:
            return
        
        try:
            total_frames = 0
            for video in self.videos:
                cap = cv2.VideoCapture(str(self.temp_folder / os.path.basename(video)))
                total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            
            cap = cv2.VideoCapture(str(self.temp_folder / os.path.basename(self.videos[0])))
            fps = self.custom_fps if self.custom_fps else cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.scale_factor)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.scale_factor)
            cap.release()
            
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                raise Exception("Can't initialize video writer.")
            
            processed_frames = 0
            self.merger_progress['maximum'] = total_frames
            for video in self.videos:
                cap = cv2.VideoCapture(str(self.temp_folder / os.path.basename(video)))
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if self.scale_factor != 1.0:
                        frame = cv2.resize(frame, (width, height))
                    out.write(frame)
                    processed_frames += 1
                    self.merger_progress['value'] = processed_frames
                    self.parent_frame.update()
                cap.release()
            
            out.release()
            self.merger_progress['value'] = 0
            tk.messagebox.showinfo("Success", "Videos merged successfully.")
        except Exception as e:
            self.merger_progress['value'] = 0
            tk.messagebox.showerror("Error", f"Merge failed: {str(e)}")

    def splitter_load_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mkv *.avi")])
        if not file_path:
            return
        try:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                tk.messagebox.showerror("Error", "Can't open video.")
                self.cap = None
                return
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                tk.messagebox.showerror("Error", "Can't get video FPS.")
                self.cap.release()
                self.cap = None
                return
            frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.duration = frame_count / fps
            self.splitter_input_entry.delete(0, tk.END)
            self.splitter_input_entry.insert(0, file_path)
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to load video: {str(e)}")
            if self.cap:
                self.cap.release()
            self.cap = None

    def splitter_split_video(self):
        if not self.cap:
            tk.messagebox.showerror("Error", "Load a video first.")
            return
        output_path = self.splitter_output_entry.get()
        if not output_path:
            tk.messagebox.showerror("Error", "Pick an output folder.")
            return
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path)
            except:
                tk.messagebox.showerror("Error", "Can't make folder.")
                return
        try:
            fps = self.custom_fps if self.custom_fps else self.cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.scale_factor)
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.scale_factor)
            
            split_value = float(self.split_value.get())
            if split_value <= 0:
                raise ValueError("Split value must be positive.")
            
            if self.split_mode.get() == "parts":
                num_parts = int(split_value)
                if num_parts <= 0:
                    raise ValueError("Number of parts must be positive.")
                frames_per_part = frame_count // num_parts
                splits = [(i * frames_per_part, (i + 1) * frames_per_part) for i in range(num_parts)]
                if frame_count % num_parts != 0:
                    splits[-1] = (splits[-1][0], frame_count)
            else:
                split_duration = split_value
                frames_per_split = int(split_duration * fps)
                splits = [(i * frames_per_split, (i + 1) * frames_per_split) for i in range(int(frame_count // frames_per_split))]
                if frame_count % frames_per_split != 0:
                    splits.append((splits[-1][1], frame_count))
            
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            for idx, (start_frame, end_frame) in enumerate(splits):
                output_file = os.path.join(output_path, f"part_{idx + 1}.mp4")
                out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
                if not out.isOpened():
                    raise Exception("Can't initialize video writer.")
                
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                for i in range(start_frame, end_frame):
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    if self.scale_factor != 1.0:
                        frame = cv2.resize(frame, (width, height))
                    out.write(frame)
                out.release()
            
            tk.messagebox.showinfo("Success", "Video split successfully.")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Split failed: {str(e)}")

    def update_codec(self):
        self.codec = self.codec_var.get()

    def update_scale(self):
        try:
            scale = float(self.scale_entry.get())
            if 0.1 <= scale <= 2.0:
                self.scale_factor = scale
                tk.messagebox.showinfo("Success", f"Scale updated to {scale}.")
            else:
                tk.messagebox.showerror("Error", "Scale must be between 0.1 and 2.0.")
        except:
            tk.messagebox.showerror("Error", "Invalid scale value.")

    def update_fps(self):
        try:
            fps = self.fps_entry.get()
            if fps.strip() == "":
                self.custom_fps = None
                tk.messagebox.showinfo("Success", "FPS reset to default.")
            else:
                fps = float(fps)
                if fps <= 0:
                    raise ValueError("FPS must be positive.")
                self.custom_fps = fps
                tk.messagebox.showinfo("Success", f"FPS updated to {fps}.")
        except:
            tk.messagebox.showerror("Error", "Invalid FPS value.")

    def browse_output(self, entry):
        folder_path = filedialog.askdirectory()
        if folder_path:
            entry.delete(0, tk.END)
            entry.insert(0, folder_path)

    def delete_temp_files(self):
        try:
            shutil.rmtree(self.temp_folder)
            self.temp_folder.mkdir(exist_ok=True)
            tk.messagebox.showinfo("Success", "Temp files deleted.")
        except:
            tk.messagebox.showerror("Error", "Can't delete temp files.")

    def cleanup(self):
        if self.cap:
            self.cap.release()
        self.delete_temp_files()

class AIAssistant:
    def __init__(self, root, config):
        self.root = root
        self.root.title("OmniCore")
        self.root.geometry("900x700")
        self.config = config
        self.model = None
        self.tokenizer = None
        self.image_pipe = None
        self.clip_model = None
        self.clip_processor = None
        self.status_var = tk.StringVar()
        self.status_var.set("Initializing...")
        self.progress_var = tk.DoubleVar()
        self.chats = self.load_chat_history()
        self.current_chat_id = None
        self.active_chat_tabs = {}
        self.chat_frames = {}
        self.chat_displays = {}
        self.input_fields = {}
        self.chat_length_vars = {}
        self.game_instance = None
        self.create_gui()
        threading.Thread(target=self.load_model, daemon=True).start()
        threading.Thread(target=self.load_image_pipeline, daemon=True).start()
        threading.Thread(target=self.load_clip_model, daemon=True).start()

    def update_status(self, message, progress=0, estimated_time=None):
        self.root.after(0, lambda: self.status_var.set(
            f"{message} {f'ETA: {estimated_time}s' if estimated_time else ''}"
        ))
        self.root.after(0, lambda: self.progress_var.set(progress))

    def simulate_progress(self, task_name, total_time):
        steps = 20
        step_time = total_time / steps
        for i in range(steps + 1):
            progress = (i / steps) * 100
            remaining_time = step_time * (steps - i)
            self.update_status(task_name, progress, round(remaining_time, 1))
            time.sleep(step_time)
        self.update_status("Ready", 100, 0)

    def load_model(self):
        eta = {"Eco": 20, "Balanced": 40, "Max": 60}[self.config["power_level"]]
        self.simulate_progress("Loading TinyLlama model...", eta)
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            self.update_status("Ready", 100, 0)
        except Exception as e:
            self.update_status(f"Error loading model: {str(e)}", 0)
            self.root.after(0, lambda: tk.messagebox.showerror("Error", "Failed to load chat model."))

    def load_image_pipeline(self):
        eta = {"Eco": 40, "Balanced": 80, "Max": 120}[self.config["power_level"]]
        self.simulate_progress("Loading Stable Diffusion model...", eta)
        try:
            self.image_pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32,
                use_auth_token=False
            )
            self.image_pipe = self.image_pipe.to("cpu")
            self.update_status("Ready", 100, 0)
        except Exception as e:
            self.update_status(f"Error loading image pipeline: {str(e)}", 0)
            self.root.after(0, lambda: tk.messagebox.showerror("Error", "Failed to load image model."))

    def load_clip_model(self):
        eta = {"Eco": 15, "Balanced": 30, "Max": 45}[self.config["power_level"]]
        self.simulate_progress("Loading CLIP model...", eta)
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.update_status("Ready", 100, 0)
        except Exception as e:
            self.update_status(f"Error loading CLIP model: {str(e)}", 0)
            self.root.after(0, lambda: tk.messagebox.showerror("Error", "Failed to load vision model."))

    def load_chat_history(self):
        self.simulate_progress("Loading chat history...", 1)
        with open(CHAT_HISTORY_FILE, 'r') as f:
            chats = json.load(f)["chats"]
        self.update_status("Ready", 100, 0)
        return chats

    def save_chat_history(self):
        self.simulate_progress("Saving chat history...", 1)
        with open(CHAT_HISTORY_FILE, 'w') as f:
            json.dump({"chats": self.chats}, f, indent=4)
        self.update_status("Ready", 100, 0)

    def create_gui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=5, fill=tk.X)
        tk.Button(top_frame, text="New Chat", command=self.new_chat).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Delete Everything", command=self.delete_everything).pack(side=tk.LEFT, padx=5)
        self.chat_list = tk.Listbox(top_frame, width=50, height=5)
        self.chat_list.pack(side=tk.LEFT, padx=5)
        self.chat_list.bind('<<ListboxSelect>>', self.load_chat)
        self.update_chat_list()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.settings_frame = tk.Frame(self.notebook)
        self.notebook.add(self.settings_frame, text="Settings")
        self.create_settings_tab()

        self.video_frame = tk.Frame(self.notebook)
        self.notebook.add(self.video_frame, text="Video Editor")
        self.video_app = VideoApp(self.video_frame)

        self.games_frame = tk.Frame(self.notebook)
        self.notebook.add(self.games_frame, text="Games")
        self.create_games_tab()

        self.add_chat_tab("Chat 1")
        self.notebook.select(self.chat_frames["Chat 1"])

        status_frame = tk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        status_label = tk.Label(status_frame, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(fill=tk.X)
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X)

    def create_settings_tab(self):
        tk.Label(self.settings_frame, text="Performance Mode:").pack(pady=5)
        self.performance_var = tk.StringVar(value=self.config["performance_mode"])
        frame = tk.Frame(self.settings_frame)
        frame.pack()
        tk.Radiobutton(frame, text="Low PC Mode", variable=self.performance_var, value="Low").pack(side=tk.LEFT)
        tk.Radiobutton(frame, text="High Performance", variable=self.performance_var, value="High").pack(side=tk.LEFT)
        tk.Button(frame, text="?", command=lambda: tk.messagebox.showinfo("Performance Mode", "Low PC Mode reduces resource usage. High Performance uses more resources for better quality.")).pack(side=tk.LEFT, padx=5)

        tk.Label(self.settings_frame, text="Image Generation Quality:").pack(pady=5)
        self.image_quality_var = tk.StringVar(value=self.config["image_quality"])
        frame = tk.Frame(self.settings_frame)
        frame.pack()
        tk.Radiobutton(frame, text="Low (faster)", variable=self.image_quality_var, value="Low").pack(side=tk.LEFT)
        tk.Radiobutton(frame, text="High (better quality)", variable=self.image_quality_var, value="High").pack(side=tk.LEFT)
        tk.Button(frame, text="?", command=lambda: tk.messagebox.showinfo("Image Quality", "Low: ~15s. High: ~30s, better quality.")).pack(side=tk.LEFT, padx=5)

        tk.Label(self.settings_frame, text="Power Level:").pack(pady=5)
        self.power_var = tk.StringVar(value=self.config["power_level"])
        frame = tk.Frame(self.settings_frame)
        frame.pack()
        tk.Radiobutton(frame, text="Eco", variable=self.power_var, value="Eco").pack(side=tk.LEFT)
        tk.Radiobutton(frame, text="Balanced", variable=self.power_var, value="Balanced").pack(side=tk.LEFT)
        tk.Radiobutton(frame, text="Max", variable=self.power_var, value="Max").pack(side=tk.LEFT)
        tk.Button(frame, text="?", command=lambda: tk.messagebox.showinfo("Power Level", "Eco: Minimal resources. Balanced: Good performance. Max: Best quality, high resource use.")).pack(side=tk.LEFT, padx=5)

        tk.Button(self.settings_frame, text="Save Settings", command=self.save_settings).pack(pady=10)

    def create_games_tab(self):
        self.games_canvas = tk.Canvas(self.games_frame, width=400, height=400, bg="black")
        self.games_canvas.pack(pady=10)
        tk.Button(self.games_frame, text="Start Snake Game", command=self.start_snake_game).pack(pady=5)
        tk.Label(self.games_frame, text="Controls: WASD/Arrow Keys to move, Q/E/Click to restart, ESC to exit").pack(pady=5)

    def start_snake_game(self):
        self.game_instance = SnakeGame(self.games_canvas, self.exit_game)
        self.games_canvas.after(100, self.game_instance.update)

    def exit_game(self):
        self.game_instance = None
        self.games_canvas.delete("all")
        self.games_canvas.create_text(
            200, 200, text="Select a game to play", fill="white", justify="center"
        )

    def save_settings(self):
        self.config["performance_mode"] = self.performance_var.get()
        self.config["image_quality"] = self.image_quality_var.get()
        self.config["power_level"] = self.power_var.get()
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=4)
        self.root.after(0, lambda: tk.messagebox.showinfo("Settings Saved", "Settings have been saved."))

    def add_chat_tab(self, tab_name):
        frame = tk.Frame(self.notebook)
        self.notebook.add(frame, text=tab_name)
        self.chat_frames[tab_name] = frame

        chat_display = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=20, state='disabled')
        chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_displays[tab_name] = chat_display

        input_frame = tk.Frame(frame)
        input_frame.pack(pady=5, fill=tk.X)
        
        tk.Label(input_frame, text="Chat Length:").pack(side=tk.LEFT, padx=5)
        self.chat_length_vars[tab_name] = tk.StringVar(value=self.config["max_chat_length"])
        tk.Radiobutton(input_frame, text="Short", variable=self.chat_length_vars[tab_name], value="Short").pack(side=tk.LEFT)
        tk.Radiobutton(input_frame, text="Long", variable=self.chat_length_vars[tab_name], value="Long").pack(side=tk.LEFT)
        
        input_field = tk.Entry(input_frame)
        input_field.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        input_field.bind('<Return>', lambda event, tn=tab_name: self.process_input(tn))
        self.input_fields[tab_name] = input_field
        
        tk.Button(input_frame, text="Send", command=lambda tn=tab_name: self.process_input(tn)).pack(side=tk.LEFT, padx=5)
        tk.Button(input_frame, text="Attach", command=lambda tn=tab_name: self.attach_file(tn)).pack(side=tk.LEFT, padx=5)
        tk.Button(input_frame, text="Generate Image", command=lambda tn=tab_name: self.generate_image(tn)).pack(side=tk.LEFT, padx=5)

    def update_chat_list(self):
        self.chat_list.delete(0, tk.END)
        for chat_id in self.chats:
            self.chat_list.insert(tk.END, f"Chat {chat_id} - {self.chats[chat_id]['timestamp']}")

    def new_chat(self):
        self.simulate_progress("Creating new chat...", 1)
        chat_id = str(len(self.chats) + 1)
        self.chats[chat_id] = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": []
        }
        self.current_chat_id = chat_id
        self.save_chat_history()
        self.update_chat_list()
        tab_name = f"Chat {len(self.active_chat_tabs) + 1}"
        self.add_chat_tab(tab_name)
        self.active_chat_tabs[tab_name] = chat_id
        self.notebook.select(self.chat_frames[tab_name])
        self.chat_displays[tab_name].config(state='normal')
        self.chat_displays[tab_name].delete(1.0, tk.END)
        self.chat_displays[tab_name].config(state='disabled')
        self.update_status("Ready", 100, 0)

    def load_chat(self, event):
        self.simulate_progress("Loading chat...", 1)
        selection = self.chat_list.curselection()
        if not selection:
            self.update_status("Ready", 100, 0)
            return
        chat_id = self.chat_list.get(selection[0]).split(" - ")[0].replace("Chat ", "")
        for tab_name, open_chat_id in self.active_chat_tabs.items():
            if open_chat_id == chat_id:
                self.notebook.select(self.chat_frames[tab_name])
                self.current_chat_id = chat_id
                self.update_status("Ready", 100, 0)
                return
        tab_name = f"Chat {len(self.active_chat_tabs) + 1}"
        self.add_chat_tab(tab_name)
        self.active_chat_tabs[tab_name] = chat_id
        self.notebook.select(self.chat_frames[tab_name])
        self.current_chat_id = chat_id
        chat_display = self.chat_displays[tab_name]
        chat_display.config(state='normal')
        chat_display.delete(1.0, tk.END)
        for msg in self.chats[chat_id]["messages"]:
            role = msg["role"]
            content = msg["content"]
            chat_display.insert(tk.END, f"{role}: {content}\n\n")
        chat_display.config(state='disabled')
        self.update_status("Ready", 100, 0)

    def process_input(self, tab_name):
        if not self.current_chat_id:
            self.root.after(0, lambda: tk.messagebox.showwarning("No Chat", "Please start a new chat first."))
            return
        input_field = self.input_fields[tab_name]
        user_input = input_field.get().strip()
        if not user_input:
            return
        self.display_message(tab_name, "User", user_input)
        self.simulate_progress("Processing query...", 10)
        response = self.process_query(user_input, tab_name)
        self.display_message(tab_name, "AI", response)
        self.chats[self.current_chat_id]["messages"].extend([
            {"role": "User", "content": user_input},
            {"role": "AI", "content": response}
        ])
        self.save_chat_history()
        input_field.delete(0, tk.END)

    def display_message(self, tab_name, role, content):
        chat_display = self.chat_displays[tab_name]
        chat_display.config(state='normal')
        chat_display.insert(tk.END, f"{role}: {content}\n\n")
        chat_display.config(state='disabled')
        chat_display.yview(tk.END)

    def attach_file(self, tab_name):
        if not self.current_chat_id:
            self.root.after(0, lambda: tk.messagebox.showwarning("No Chat", "Please start a new chat first."))
            return
        file_path = filedialog.askopenfilename(filetypes=[
            ("All supported", "*.txt;*.pdf;*.png;*.jpg;*.jpeg"),
            ("Text files", "*.txt"),
            ("PDF files", "*.pdf"),
            ("Image files", "*.png;*.jpg;*.jpeg")
        ])
        if not file_path:
            return
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in [".exe", ".bat", ".sh"]:
            self.root.after(0, lambda: tk.messagebox.showerror("Security Error", "Executable files not allowed."))
            return
        self.simulate_progress("Saving attachment...", 2)
        file_name = os.path.basename(file_path).replace("..", "").replace("/", "").replace("\\", "")
        dest_path = os.path.join(ATTACHMENTS_DIR, file_name)
        with open(file_path, 'rb') as src, open(dest_path, 'wb') as dst:
            dst.write(src.read())
        response = self.process_attachment(file_path, tab_name)
        self.display_message(tab_name, "User", f"Uploaded: {file_name}")
        self.display_message(tab_name, "AI", response)
        self.chats[self.current_chat_id]["messages"].extend([
            {"role": "User", "content": f"Uploaded: {file_name}"},
            {"role": "AI", "content": response}
        ])
        self.save_chat_history()

    def process_query(self, query, tab_name):
        if "search" in query.lower():
            return self.deep_search(query, tab_name)
        elif "think" in query.lower() or "reason" in query.lower():
            return self.deep_think(query, tab_name)
        else:
            return self.get_model_response(query, tab_name)

    def get_model_response(self, query, tab_name):
        if not self.model or not self.tokenizer:
            self.update_status("Ready", 100, 0)
            return "Model not loaded yet."
        try:
            inputs = self.tokenizer(f"<|user|> {query} <|assistant|> ", return_tensors="pt").to("cpu")
            max_length = 150 if self.chat_length_vars[tab_name].get() == "Short" else 300
            outputs = self.model.generate(**inputs, max_length=max_length, num_return_sequences=1, temperature=0.7, do_sample=True)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.update_status("Ready", 100, 0)
            return response.replace("<|user|> " + query + " <|assistant|> ", "").strip()
        except Exception as e:
            self.update_status("Ready", 100, 0)
            return f"Error processing query: {str(e)}"

    def deep_search(self, query, tab_name):
        self.simulate_progress("Performing deep search...", 15)
        prompt = f"<|user|> Perform a deep search for: {query}. Analyze and summarize findings as if searching real-time data. <|assistant|> "
        return self.get_model_response(prompt, tab_name)

    def deep_think(self, query, tab_name):
        self.simulate_progress("Deep thinking...", 20)
        prompt = f"<|user|> Analyze and reason deeply about: {query}. Break down the problem step-by-step, consider multiple approaches, and provide a detailed, reasoned answer. <|assistant|> "
        return self.get_model_response(prompt, tab_name)

    def process_attachment(self, file_path, tab_name):
        self.simulate_progress("Processing attachment...", 2)
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == ".txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.update_status("Ready", 100, 0)
            return f"Processed TXT file: {os.path.basename(file_path)}\nContent preview: {content[:100]}..."
        elif file_ext == ".pdf":
            try:
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                self.update_status("Ready", 100, 0)
                return f"Processed PDF file: {os.path.basename(file_path)}\nContent preview: {text[:100]}..."
            except Exception as e:
                self.update_status("Ready", 100, 0)
                return f"Error processing PDF: {str(e)}"
        elif file_ext in [".png", ".jpg", ".jpeg"]:
            img = Image.open(file_path)
            img.thumbnail((200, 200))
            photo = ImageTk.PhotoImage(img)
            self.chat_displays[tab_name].image_create(tk.END, image=photo)
            self.chat_displays[tab_name].insert(tk.END, "\n")
            self.chat_displays[tab_name].image = photo
            if self.clip_model and self.clip_processor:
                try:
                    image = Image.open(file_path).convert("RGB")
                    inputs = self.clip_processor(images=image, return_tensors="pt")
                    image_features = self.clip_model.get_image_features(**inputs)
                    description = self.get_model_response(
                        f"Describe this image based on its features: {image_features.tolist()[:10]}",
                        tab_name
                    )
                    self.update_status("Ready", 100, 0)
                    return f"Processed image file: {os.path.basename(file_path)}\nImage displayed above.\nDescription: {description}"
                except Exception as e:
                    self.update_status("Ready", 100, 0)
                    return f"Processed image file: {os.path.basename(file_path)}\nImage displayed above.\nError analyzing image: {str(e)}"
            else:
                self.update_status("Ready", 100, 0)
                return f"Processed image file: {os.path.basename(file_path)}\nImage displayed above.\nVision model not loaded."
        else:
            self.update_status("Ready", 100, 0)
            return "Unsupported file type."

    def generate_image(self, tab_name):
        if not self.current_chat_id:
            self.root.after(0, lambda: tk.messagebox.showwarning("No Chat", "Please start a new chat first."))
            return
        input_field = self.input_fields[tab_name]
        prompt = input_field.get().strip()
        if not prompt:
            self.root.after(0, lambda: tk.messagebox.showwarning("No Prompt", "Please enter a prompt to generate an image."))
            return
        if not self.image_pipe:
            self.root.after(0, lambda: tk.messagebox.showwarning("Image Model", "Image generation model not loaded yet."))
            return
        eta = 15 if self.config["image_quality"] == "Low" else 30
        steps = 10 if self.config["image_quality"] == "Low" else 20
        self.simulate_progress("Generating image...", eta)
        try:
            image = self.image_pipe(prompt, num_inference_steps=steps).images[0]
            img_path = os.path.join(GENERATED_IMAGES_DIR, f"generated_{len(os.listdir(GENERATED_IMAGES_DIR)) + 1}.png")
            image.save(img_path)
            img = Image.open(img_path)
            img.thumbnail((200, 200))
            photo = ImageTk.PhotoImage(img)
            self.display_message(tab_name, "AI", f"Generated image for prompt: {prompt}")
            self.chat_displays[tab_name].image_create(tk.END, image=photo)
            self.chat_displays[tab_name].insert(tk.END, "\n")
            self.chat_displays[tab_name].image = photo
            self.chats[self.current_chat_id]["messages"].extend([
                {"role": "User", "content": f"Generate image: {prompt}"},
                {"role": "AI", "content": f"Generated image for prompt: {prompt}"}
            ])
            self.save_chat_history()
        except Exception as e:
            self.display_message(tab_name, "AI", f"Error generating image: {str(e)}")
            self.update_status("Ready", 100, 0)

    def delete_everything(self):
        def perform_delete():
            self.simulate_progress("Deleting all data...", 2)
            self.chats = {}
            self.current_chat_id = None
            self.active_chat_tabs = {}
            for tab_name in list(self.chat_frames.keys()):
                self.notebook.forget(self.chat_frames[tab_name])
            self.chat_frames = {}
            self.chat_displays = {}
            self.input_fields = {}
            self.chat_length_vars = {}
            self.settings_frame = tk.Frame(self.notebook)
            self.notebook.add(self.settings_frame, text="Settings")
            self.create_settings_tab()
            self.video_frame = tk.Frame(self.notebook)
            self.notebook.add(self.video_frame, text="Video Editor")
            self.video_app = VideoApp(self.video_frame)
            self.games_frame = tk.Frame(self.notebook)
            self.notebook.add(self.games_frame, text="Games")
            self.create_games_tab()
            self.add_chat_tab("Chat 1")
            self.notebook.select(self.chat_frames["Chat 1"])
            self.save_chat_history()
            self.update_chat_list()
            self.chat_displays["Chat 1"].config(state='normal')
            self.chat_displays["Chat 1"].delete(1.0, tk.END)
            self.chat_displays["Chat 1"].config(state='disabled')
            if os.path.exists(ATTACHMENTS_DIR):
                shutil.rmtree(ATTACHMENTS_DIR)
            if os.path.exists(GENERATED_IMAGES_DIR):
                shutil.rmtree(GENERATED_IMAGES_DIR)
            if os.path.exists(VIDEO_TEMP_DIR):
                shutil.rmtree(VIDEO_TEMP_DIR)
            os.makedirs(ATTACHMENTS_DIR, exist_ok=True)
            os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)
            os.makedirs(VIDEO_TEMP_DIR, exist_ok=True)
            with open(CHAT_HISTORY_FILE, 'w') as f:
                json.dump({"chats": {}}, f)
            self.update_status("Ready", 100, 0)
            self.root.after(0, lambda: tk.messagebox.showinfo("Success", "All data deleted."))

        result = tk.messagebox.askyesno("Confirm Delete", "Delete all chats, attachments, images, and video temp files?")
        if result:
            self.root.after(0, perform_delete)

def start_main_app(config):
    root = tkdnd.TkinterDnD.Tk() if TKDND_AVAILABLE else tk.Tk()
    app = AIAssistant(root, config)
    root.mainloop()

if __name__ == "__main__":
    root = tkdnd.TkinterDnD.Tk() if TKDND_AVAILABLE else tk.Tk()
    settings_gui = SettingsGUI(root, start_main_app)
    root.mainloop()
