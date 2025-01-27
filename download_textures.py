import requests
from PIL import Image
from io import BytesIO
import os

def download_and_process_textures():
    # NASA moon texture URLs
    texture_url = "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/lroc_color_poles_1k.jpg"
    bump_url = "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/ldem_3_8bit.jpg"
    
    # Create textures directory if it doesn't exist
    os.makedirs("static/textures", exist_ok=True)
    
    # Download and save texture map
    print("Downloading texture map...")
    response = requests.get(texture_url)
    img = Image.open(BytesIO(response.content))
    img.save("static/textures/moon_texture.jpg", quality=95)
    
    # Download and save bump map
    print("Downloading bump map...")
    response = requests.get(bump_url)
    img = Image.open(BytesIO(response.content))
    img.save("static/textures/moon_bump.jpg", quality=95)
    
    print("Textures downloaded successfully!")

if __name__ == "__main__":
    download_and_process_textures() 