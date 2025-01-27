import requests
from PIL import Image
from io import BytesIO
import os

def download_and_process_textures():
    # URLs for textures
    texture_url = "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/lroc_color_poles_1k.jpg"
    bump_url = "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/ldem_3_8bit.jpg"
    night_sky_url = "https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/equirectangular/starmap_4k.hdr"
    water_normal_url = "https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/waternormals.jpg"
    
    # Create textures directory if it doesn't exist
    os.makedirs("static/textures", exist_ok=True)
    
    # Download and save texture map
    print("Downloading moon texture map...")
    response = requests.get(texture_url)
    img = Image.open(BytesIO(response.content))
    img.save("static/textures/moon_texture.jpg", quality=95)
    
    # Download and save bump map
    print("Downloading moon bump map...")
    response = requests.get(bump_url)
    img = Image.open(BytesIO(response.content))
    img.save("static/textures/moon_bump.jpg", quality=95)
    
    # Download and save night sky HDR
    print("Downloading night sky HDR...")
    response = requests.get(night_sky_url)
    with open("static/textures/night_sky.hdr", 'wb') as f:
        f.write(response.content)
    
    # Download and save water normal map
    print("Downloading water normal map...")
    response = requests.get(water_normal_url)
    img = Image.open(BytesIO(response.content))
    img.save("static/textures/waternormals.jpg", quality=95)
    
    print("Textures downloaded successfully!")

if __name__ == "__main__":
    download_and_process_textures() 