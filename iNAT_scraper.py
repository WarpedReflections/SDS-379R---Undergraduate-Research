import os
import requests
import pandas as pd 
from selenium import webdriver 
from selenium.webdriver import Chrome 
from selenium.webdriver.common.by import By 

# Function to download images from observation URLs.
def download_images(observation_id, observation_url, output_root_folder):
    # Define the Chrome webdriver options.
    options = webdriver.ChromeOptions()
    options.add_argument('--headless') # Set the Chrome webdriver to run in headless mode for scalability.

    # By default, Selenium waits for all resources to download before taking actions.
    # However, we don't need it as the page is populated with dynamically generated JavaScript code.
    options.page_load_strategy = 'none'

    # Pass the defined options objects to initialize the web driver.
    driver = Chrome(options = options)

    # Set an implicit wait of 5 seconds to allow time for elements to appear before throwing an exception.
    driver.implicitly_wait(5)

    driver.get(observation_url)

    # Find all images within the div with class 'image-gallery-image'.
    image_elements = driver.find_elements(By.CSS_SELECTOR, "div[class*='image-gallery-image'] img")

    # Create a folder for the current observation ID.
    output_folder = os.path.join(output_root_folder, f'iNAT{observation_id}')
    os.makedirs(output_folder, exist_ok = True)

    # Download and save each image.
    for index, img_element in enumerate(image_elements):
        img_url = img_element.get_attribute('src')

        # Generate a unique filename for each image with the original file extension.
        img_filename = f'iNAT{observation_id}_{index + 1}.jpg'

        # Save the image to the output folder.
        img_path = os.path.join(output_folder, img_filename)
        img_data = requests.get(img_url).content
        with open(img_path, 'wb') as img_file:
            img_file.write(img_data)

        print(f"Image downloaded: {img_filename}")

# Read the Excel file into a DataFrame.
excel_file_path = 'AttaAlates_iNAT.xlsx'
df = pd.read_excel(excel_file_path)

# Output root folder for downloaded images.
output_root_folder = 'iNAT_images'

# Create the output root folder if it doesn't exist.
os.makedirs(output_root_folder, exist_ok = True)

# Iterate through each row in the DataFrame.
for index, row in df.iterrows():
    observation_id = row["iNAT ##"]
    observation_url = row["Observation URL"]

    # Download images only if the observation URL is available.
    if pd.notna(observation_url):
        download_images(observation_id, observation_url, output_root_folder)

print("All images downloaded successfully.")