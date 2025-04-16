import subprocess
import sys

def upgrade_pip():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("Pip upgraded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error upgrading pip: {e}")
        sys.exit(1)

def install_packages():
    packages = [
        "transformers==4.45.2",
        "torch==2.5.0",
        "PyPDF2==3.0.1",
        "accelerate==0.34.2",
        "diffusers==0.30.3"
    ]
    for package in packages:
        print(f"Installing {package}...")
        try:
            cmd = [sys.executable, "-m", "pip", "install", package]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error installing {package}:\n{result.stderr}")
                sys.exit(1)
            print(f"{package} installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
            sys.exit(1)

if __name__ == "__main__":
    upgrade_pip()
    install_packages()
    print("All requirements installed. Run 'python install_tinyllama.py' to download the TinyLlama model, then 'python main.py' to start OmniCore.")