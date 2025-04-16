import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import json
import os
import shutil
from PIL import Image, ImageTk
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import PyPDF2
import threading
import time
from diffusers import StableDiffusionPipeline

ATTACHMENTS_DIR = "attachments"
GENERATED_IMAGES_DIR = "generated_images"
CHAT_HISTORY_FILE = "chat_history.json"
CONFIG_FILE = "config.json"

os.makedirs(ATTACHMENTS_DIR, exist_ok=True)
os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)

# Ensure chat_history.json is initialized properly
def initialize_chat_history():
    default_history = {"chats": {}}
    if not os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'w') as f:
            json.dump(default_history, f)
    else:
        # Validate and fix corrupted file
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
        self.root.geometry("400x400")
        self.callback = callback
        self.config = self.load_config()
        self.create_gui()

    def load_config(self):
        default_config = {
            "performance_mode": "High",
            "image_quality": "High",
            "max_chat_length": "Long"
        }
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        return default_config

    def save_config(self):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=4)

    def create_gui(self):
        # Performance Mode
        tk.Label(self.root, text="Performance Mode:").pack(pady=5)
        self.performance_var = tk.StringVar(value=self.config["performance_mode"])
        frame = tk.Frame(self.root)
        frame.pack()
        tk.Radiobutton(frame, text="Low PC Mode", variable=self.performance_var, value="Low").pack(side=tk.LEFT)
        tk.Button(frame, text="?", command=lambda: messagebox.showinfo("Performance Mode", "Low PC Mode reduces resource usage for slower PCs. High Performance Mode uses more resources for better quality.")).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(frame, text="High Performance", variable=self.performance_var, value="High").pack(side=tk.LEFT)

        # Image Quality
        tk.Label(self.root, text="Image Generation Quality:").pack(pady=5)
        self.image_quality_var = tk.StringVar(value=self.config["image_quality"])
        frame = tk.Frame(self.root)
        frame.pack()
        tk.Radiobutton(frame, text="Low (faster)", variable=self.image_quality_var, value="Low").pack(side=tk.LEFT)
        tk.Radiobutton(frame, text="High (better quality)", variable=self.image_quality_var, value="High").pack(side=tk.LEFT)
        tk.Button(frame, text="?", command=lambda: messagebox.showinfo("Image Quality", "Low quality generates images faster (~15s). High quality takes longer (~30s) but looks better.")).pack(side=tk.LEFT, padx=5)

        # Max Chat Length
        tk.Label(self.root, text="Max Chat Length:").pack(pady=5)
        self.chat_length_var = tk.StringVar(value=self.config["max_chat_length"])
        frame = tk.Frame(self.root)
        frame.pack()
        tk.Radiobutton(frame, text="Short (faster)", variable=self.chat_length_var, value="Short").pack(side=tk.LEFT)
        tk.Radiobutton(frame, text="Long (more detailed)", variable=self.chat_length_var, value="Long").pack(side=tk.LEFT)
        tk.Button(frame, text="?", command=lambda: messagebox.showinfo("Max Chat Length", "Short responses (~150 words) are faster. Long responses (~300 words) are more detailed but slower.")).pack(side=tk.LEFT, padx=5)

        # Start Button
        tk.Button(self.root, text="Start OmniCore", command=self.start).pack(pady=20)

    def start(self):
        self.config["performance_mode"] = self.performance_var.get()
        self.config["image_quality"] = self.image_quality_var.get()
        self.config["max_chat_length"] = self.chat_length_var.get()
        self.save_config()
        self.root.destroy()
        self.callback(self.config)

class AIAssistant:
    def __init__(self, root, config):
        self.root = root
        self.root.title("OmniCore")
        self.root.geometry("800x600")
        self.config = config
        self.model = None
        self.tokenizer = None
        self.image_pipe = None
        self.status_var = tk.StringVar()
        self.status_var.set("Initializing...")
        self.progress_var = tk.DoubleVar()
        self.chats = self.load_chat_history()
        self.current_chat_id = None
        self.active_chat_tabs = {}
        self.chat_frames = {}
        self.chat_displays = {}
        self.input_fields = {}
        self.create_gui()
        threading.Thread(target=self.load_model, daemon=True).start()
        threading.Thread(target=self.load_image_pipeline, daemon=True).start()

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
        eta = 60 if self.config["performance_mode"] == "High" else 30
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
            self.root.after(0, lambda: messagebox.showerror("Error", "Failed to load chat model. Check your internet connection and try again."))

    def load_image_pipeline(self):
        eta = 120 if self.config["performance_mode"] == "High" else 60
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
            self.root.after(0, lambda: messagebox.showerror("Error", "Failed to load image model. Check your internet connection and disk space (~4GB needed)."))

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
        # Top frame with buttons and chat list
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=5, fill=tk.X)
        tk.Button(top_frame, text="New Chat", command=self.new_chat).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Delete Everything", command=self.delete_everything).pack(side=tk.LEFT, padx=5)
        self.chat_list = tk.Listbox(top_frame, width=50, height=5)
        self.chat_list.pack(side=tk.LEFT, padx=5)
        self.chat_list.bind('<<ListboxSelect>>', self.load_chat)
        self.update_chat_list()

        # Notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Settings Tab (First Tab)
        self.settings_frame = tk.Frame(self.notebook)
        self.notebook.add(self.settings_frame, text="Settings")
        self.create_settings_tab()

        # Default Chat Tab (Second Tab)
        self.add_chat_tab("Chat 1")
        self.notebook.select(self.chat_frames["Chat 1"])  # Ensure Chat 1 is selected

        # Status bar with progress
        status_frame = tk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        status_label = tk.Label(status_frame, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(fill=tk.X)
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X)

    def create_settings_tab(self):
        # Performance Mode
        tk.Label(self.settings_frame, text="Performance Mode:").pack(pady=5)
        self.performance_var = tk.StringVar(value=self.config["performance_mode"])
        frame = tk.Frame(self.settings_frame)
        frame.pack()
        tk.Radiobutton(frame, text="Low PC Mode", variable=self.performance_var, value="Low").pack(side=tk.LEFT)
        tk.Button(frame, text="?", command=lambda: messagebox.showinfo("Performance Mode", "Low PC Mode reduces resource usage for slower PCs. High Performance Mode uses more resources for better quality.")).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(frame, text="High Performance", variable=self.performance_var, value="High").pack(side=tk.LEFT)

        # Image Quality
        tk.Label(self.settings_frame, text="Image Generation Quality:").pack(pady=5)
        self.image_quality_var = tk.StringVar(value=self.config["image_quality"])
        frame = tk.Frame(self.settings_frame)
        frame.pack()
        tk.Radiobutton(frame, text="Low (faster)", variable=self.image_quality_var, value="Low").pack(side=tk.LEFT)
        tk.Radiobutton(frame, text="High (better quality)", variable=self.image_quality_var, value="High").pack(side=tk.LEFT)
        tk.Button(frame, text="?", command=lambda: messagebox.showinfo("Image Quality", "Low quality generates images faster (~15s). High quality takes longer (~30s) but looks better.")).pack(side=tk.LEFT, padx=5)

        # Max Chat Length
        tk.Label(self.settings_frame, text="Max Chat Length:").pack(pady=5)
        self.chat_length_var = tk.StringVar(value=self.config["max_chat_length"])
        frame = tk.Frame(self.settings_frame)
        frame.pack()
        tk.Radiobutton(frame, text="Short (faster)", variable=self.chat_length_var, value="Short").pack(side=tk.LEFT)
        tk.Radiobutton(frame, text="Long (more detailed)", variable=self.chat_length_var, value="Long").pack(side=tk.LEFT)
        tk.Button(frame, text="?", command=lambda: messagebox.showinfo("Max Chat Length", "Short responses (~150 words) are faster. Long responses (~300 words) are more detailed but slower.")).pack(side=tk.LEFT, padx=5)

        tk.Button(self.settings_frame, text="Save Settings", command=self.save_settings).pack(pady=10)

    def save_settings(self):
        self.config["performance_mode"] = self.performance_var.get()
        self.config["image_quality"] = self.image_quality_var.get()
        self.config["max_chat_length"] = self.chat_length_var.get()
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=4)
        self.root.after(0, lambda: messagebox.showinfo("Settings Saved", "Settings have been saved."))

    def add_chat_tab(self, tab_name):
        frame = tk.Frame(self.notebook)
        self.notebook.add(frame, text=tab_name)  # Use add instead of insert to avoid index issues
        self.chat_frames[tab_name] = frame

        chat_display = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=20, state='disabled')
        chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_displays[tab_name] = chat_display

        input_frame = tk.Frame(frame)
        input_frame.pack(pady=5, fill=tk.X)
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
            self.root.after(0, lambda: messagebox.showwarning("No Chat", "Please start a new chat first."))
            return
        input_field = self.input_fields[tab_name]
        user_input = input_field.get().strip()
        if not user_input:
            return
        self.display_message(tab_name, "User", user_input)
        self.simulate_progress("Processing query...", 10)
        response = self.process_query(user_input)
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
            self.root.after(0, lambda: messagebox.showwarning("No Chat", "Please start a new chat first."))
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
            self.root.after(0, lambda: messagebox.showerror("Security Error", "Unsupported file type: Executable files are not allowed."))
            return
        self.simulate_progress("Saving attachment...", 2)
        file_name = os.path.basename(file_path).replace("..", "").replace("/", "").replace("\\", "")
        dest_path = os.path.join(ATTACHMENTS_DIR, file_name)
        with open(file_path, 'rb') as src, open(dest_path, 'wb') as dst:
            dst.write(src.read())
        response = self.process_attachment(file_path)
        self.display_message(tab_name, "User", f"Uploaded: {file_name}")
        self.display_message(tab_name, "AI", response)
        self.chats[self.current_chat_id]["messages"].extend([
            {"role": "User", "content": f"Uploaded: {file_name}"},
            {"role": "AI", "content": response}
        ])
        self.save_chat_history()

    def process_query(self, query):
        if "search" in query.lower():
            return self.deep_search(query)
        elif "think" in query.lower() or "reason" in query.lower():
            return self.deep_think(query)
        else:
            return self.get_model_response(query)

    def get_model_response(self, query):
        if not self.model or not self.tokenizer:
            self.update_status("Ready", 100, 0)
            return "Model not loaded yet. Please wait."
        try:
            inputs = self.tokenizer(f"<|user|> {query} <|assistant|> ", return_tensors="pt").to("cpu")
            max_length = 150 if self.config["max_chat_length"] == "Short" else 300
            outputs = self.model.generate(**inputs, max_length=max_length, num_return_sequences=1, temperature=0.7, do_sample=True)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.update_status("Ready", 100, 0)
            return response.replace("<|user|> " + query + " <|assistant|> ", "").strip()
        except Exception as e:
            self.update_status("Ready", 100, 0)
            return f"Error processing query: {str(e)}"

    def deep_search(self, query):
        self.simulate_progress("Performing deep search...", 15)
        prompt = f"<|user|> Perform a deep search for: {query}. Analyze and summarize findings as if searching real-time data. <|assistant|> "
        return self.get_model_response(prompt)

    def deep_think(self, query):
        self.simulate_progress("Deep thinking...", 20)
        prompt = f"<|user|> Analyze and reason deeply about: {query}. Break down the problem step-by-step, consider multiple approaches, and provide a detailed, reasoned answer. <|assistant|> "
        return self.get_model_response(prompt)

    def process_attachment(self, file_path):
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
            self.chat_displays[list(self.active_chat_tabs.keys())[self.notebook.index(self.notebook.select())]].image_create(tk.END, image=photo)
            self.chat_displays[list(self.active_chat_tabs.keys())[self.notebook.index(self.notebook.select())]].insert(tk.END, "\n")
            self.chat_displays[list(self.active_chat_tabs.keys())[self.notebook.index(self.notebook.select())]].image = photo
            self.update_status("Ready", 100, 0)
            return f"Processed image file: {os.path.basename(file_path)}\nImage displayed above."
        else:
            self.update_status("Ready", 100, 0)
            return "Unsupported file type."

    def generate_image(self, tab_name):
        if not self.current_chat_id:
            self.root.after(0, lambda: messagebox.showwarning("No Chat", "Please start a new chat first."))
            return
        input_field = self.input_fields[tab_name]
        prompt = input_field.get().strip()
        if not prompt:
            self.root.after(0, lambda: messagebox.showwarning("No Prompt", "Please enter a prompt to generate an image."))
            return
        if not self.image_pipe:
            self.root.after(0, lambda: messagebox.showwarning("Image Model", "Image generation model not loaded yet. Please wait."))
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
            # Clear all tabs except Settings
            for tab_name in list(self.chat_frames.keys()):
                self.notebook.forget(self.chat_frames[tab_name])
            self.chat_frames = {}
            self.chat_displays = {}
            self.input_fields = {}
            # Re-add Settings tab first
            self.settings_frame = tk.Frame(self.notebook)
            self.notebook.add(self.settings_frame, text="Settings")
            self.create_settings_tab()
            # Add default Chat 1 tab
            self.add_chat_tab("Chat 1")
            self.notebook.select(self.chat_frames["Chat 1"])
            self.save_chat_history()
            self.update_chat_list()
            self.chat_displays["Chat 1"].config(state='normal')
            self.chat_displays["Chat 1"].delete(1.0, tk.END)
            self.chat_displays["Chat 1"].config(state='disabled')
            # Clear attachments and generated images
            if os.path.exists(ATTACHMENTS_DIR):
                shutil.rmtree(ATTACHMENTS_DIR)
            if os.path.exists(GENERATED_IMAGES_DIR):
                shutil.rmtree(GENERATED_IMAGES_DIR)
            os.makedirs(ATTACHMENTS_DIR, exist_ok=True)
            os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)
            # Reset chat_history.json
            with open(CHAT_HISTORY_FILE, 'w') as f:
                json.dump({"chats": {}}, f)
            self.update_status("Ready", 100, 0)
            self.root.after(0, lambda: messagebox.showinfo("Success", "All data deleted."))

        # Use askyesno and handle result directly
        result = messagebox.askyesno("Confirm Delete", "Are you sure you want to delete all chats, attachments, and generated images?")
        if result:
            self.root.after(0, perform_delete)

def start_main_app(config):
    root = tk.Tk()
    app = AIAssistant(root, config)
    root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    settings_gui = SettingsGUI(root, start_main_app)
    root.mainloop()