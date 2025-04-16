OmniCore - Your Ultimate AI Assistant

 OmniCore is a powerful AI assistant that can chat with you, generate images, and process files like PDFs and images. It’s designed to grow into an infinite-capability AI that might one day change the world! Right now, it’s a desktop app you can run on Windows with a simple double-click setup.

 ## 🚀 Easy Setup for Everyone

 ### What You Need
 - A Windows PC with:
   - At least 8GB of RAM (for smooth performance).
   - At least 10GB of free disk space (for AI models).
   - A good internet connection (to download ~6GB of AI models the first time).
 - Python installed:
   - If you don’t have Python, download and install it from [python.org](https://www.python.org/downloads/). Choose the latest version (e.g., Python 3.13), and during setup, check the box to "Add Python to PATH."

 ### Step-by-Step Setup
 1. **Download OmniCore**:
    - Click the green "Code" button on this GitHub page, then click "Download ZIP."
    - Unzip the folder to your Desktop (e.g., `C:\Users\YourName\Desktop\OmniCore`).

 2. **Install Required Libraries**:
    - Open the `OmniCore` folder on your Desktop.
    - Find the file named `install_requirements.py`.
    - Double-click `install_requirements.py` to run it.
    - A black window will pop up and install everything OmniCore needs (like `transformers` and `diffusers`). This might take a few minutes. Wait until the window closes or says “All requirements installed.”

 3. **Download the AI Chat Model**:
    - In the same folder, find `install_tinyllama.py`.
    - Double-click `install_tinyllama.py` to run it.
    - This downloads the TinyLlama-1.1B-Chat model (~2GB). It’ll take a few minutes the first time. Wait until the window says “TinyLlama-1.1B-Chat is ready.”

 4. **Start OmniCore**:
    - In the same folder, find `main.py`.
    - Double-click `main.py` to run it.
    - A settings window will pop up first. Choose your options (like Low PC Mode), then click “Start OmniCore.”
    - The main window called “OmniCore” will open, starting with the “Chat 1” tab. You’re ready to use it!

 ### How to Use OmniCore
 OmniCore has a simple interface with powerful features. Here’s what you can do:

 - **Choose Settings Before Starting**:
   - When you first run OmniCore, a settings window opens.
   - Pick “Low PC Mode” if your computer is slow, or “High Performance” for better quality.
   - Set image quality (Low for faster images, High for better quality).
   - Set chat length (Short for faster replies, Long for detailed ones).
   - Click the `?` buttons next to each option to learn what they do.

 - **Start a Chat**:
   - Click “New Chat” to start a new conversation. It opens in a new tab (e.g., “Chat 1”).
   - Type your message (e.g., “How’s it going?”) in the text box at the bottom.
   - Press Enter or click “Send” to get a response from OmniCore.
   - Responses take ~10-20 seconds (it’s thinking!).

 - **Use Multiple Chat Tabs**:
   - Each new chat opens in a new tab (e.g., “Chat 2”).
   - Click a chat in the list to open it in a new tab.
   - You can’t open the same chat in multiple tabs.
   - Tabs activate when you start or select a chat.

 - **Generate Images**:
   - Type a description (e.g., “a sunny beach”) in the text box.
   - Click “Generate Image” to create a picture based on your description.
   - Image generation takes ~15-30 seconds (depending on your settings). You’ll see the image in the chat.

 - **Attach Files**:
   - Click “Attach” to upload a file (TXT, PDF, PNG, JPG, JPEG).
   - OmniCore will process the file and show a preview in the chat (e.g., text from a PDF or the image itself).

 - **Adjust Settings Anytime**:
   - Click the “Settings” tab to change options like performance mode.
   - Click the `?` buttons to learn what each setting does.
   - Click “Save Settings” to apply changes.

 - **Delete Everything**:
   - Click “Delete Everything” to clear all chats, attachments, and generated images.
   - Confirm by clicking “Yes” in the popup.

 - **Status Bar**:
   - At the bottom, you’ll see a status bar with a progress bar and time estimate (ETA).
   - It shows what OmniCore is doing (e.g., “Loading TinyLlama model… ETA: 60s” or “Generating image… ETA: 30s”).

 ### Tips
 - The first time you run OmniCore, it downloads two AI models: TinyLlama (~2GB) for chatting and Stable Diffusion (~4GB) for images. This takes ~5-10 minutes.
 - If OmniCore is slow or crashes, close other apps to free up memory, or check if you have enough disk space.
 - If the GUI doesn’t open, double-check that Python is installed and you ran `install_requirements.py`.

 ---

 ## 🛠️ Technical Details for Developers

 ### Project Structure
 - **`main.py`**: The core application with the pre-launch settings GUI, tabbed chat interface, settings tab, chat logic, image generation, and file processing.
 - **`install_requirements.py`**: Installs dependencies with pinned versions (`transformers==4.45.2`, `torch==2.5.0`, `PyPDF2==3.0.1`, `accelerate==0.34.2`, `diffusers==0.30.3`).
 - **`install_tinyllama.py`**: Downloads the TinyLlama-1.1B-Chat model (~2GB) for chat functionality.
 - **`chat_history.json`**: Stores chat history in JSON format.
 - **`config.json`**: Stores user settings (performance mode, etc.).
 - **`attachments/`**: Folder for uploaded files (TXT, PDF, images).
 - **`generated_images/`**: Folder for AI-generated images (PNG).

 ### Dependencies
 - Python 3.13 (tested on Windows).
 - Libraries (installed via `install_requirements.py`):
   - `transformers==4.45.2`: For TinyLlama-1.1B-Chat (chat model).
   - `torch==2.5.0`: For model inference (CPU-only).
   - `PyPDF2==3.0.1`: For PDF processing.
   - `accelerate==0.34.2`: Optimizes model loading.
   - `diffusers==0.30.3`: For Stable Diffusion (image generation).

 ### Features and Implementation
 - **Pre-Launch Settings GUI**: Built with `tkinter`, allows users to configure performance mode, image quality, and chat length. Saves to `config.json`.
 - **Tabbed Interface**: Uses `ttk.Notebook` with a “Settings” tab and dynamic chat tabs. Prevents duplicate chats in tabs.
 - **Settings Tab**: Mirrors pre-launch settings with tooltips for clarity.
 - **Chat**: Uses TinyLlama-1.1B-Chat (~2GB) with dynamic `max_length` (150 or 300 based on settings). Supports regular queries, deep thinking, and deep search.
 - **Image Generation**: Uses Stable Diffusion (runwayml/stable-diffusion-v1-5, ~4GB) with dynamic inference steps (10 or 20 based on settings). Runs on CPU.
 - **File Processing**: Handles TXT, PDF, and images with security checks (blocks `.exe`, sanitizes filenames).
 - **Status Bar**: Uses `ttk.Progressbar` and `StringVar` to show progress and ETA for all tasks (model loading, image gen, query processing, file ops). ETA is simulated.

 ### Known Limitations
 - **Performance**: CPU-only mode makes image generation slow (~15-30s). Needs optimization for low-RAM systems (<8GB).
 - **Cross-Platform**: Tested only on Windows. File paths may need adjustments for Linux/Mac (use `os.path`).
 - **Error Handling**: Improved with network error messages, but real-time progress for the status bar is still simulated.
 - **Security**: Basic checks for attachments added, but more robust validation could be implemented.

 ### Development Setup
 1. Clone the repo:
    - Download the ZIP from GitHub and unzip it.
 2. Install dependencies:
    - Double-click `install_requirements.py` to install libraries.
 3. Download the chat model:
    - Double-click `install_tinyllama.py` to download TinyLlama-1.1B-Chat.
 4. Run the app:
    - Double-click `main.py` to start the GUI.

 ### Future Improvements
 - Implement real-time progress for the status bar.
 - Add a “Cancel” button for long tasks (e.g., image generation).
 - Test and optimize for Linux/Mac (adjust file paths with `os.path`).
 - Add more advanced security checks for attachments.
 - Support newer Python versions with CUDA if needed.

 ### Contributing
 This is a beta release—tested on Windows, CPU-only. Contributions are welcome! Please test on Linux/Mac, optimize for low-spec systems, or add new features like voice input. Open an issue or submit a pull request.

 ---
