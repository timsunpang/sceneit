import os
import pandas as pd
import requests
from PIL import Image

# Define the directory to store downloaded files
output_directory = 'downloaded_posters'
os.makedirs(output_directory, exist_ok=True)

# Read the CSV file
csv_file = 'posters.csv'
df = pd.read_csv(csv_file)

# Assuming the CSV has a column named 'url' with the URLs
for index, row in df.iterrows():
    url = row['Poster']
    title = row['Title']
    title = title.replace('/', '_')
    try:
        # Get the file name from the URL
        file_name = os.path.basename(title)
        file_path = os.path.join(output_directory, file_name + '.jpg')

        # Download the file
        response = requests.get(url)
        response.raise_for_status()  # Check for request errors

        # Write the content to a file
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded: {file_name}")

    except (requests.exceptions.RequestException, TypeError) as e:
        print(f"Failed to download {url}: {e}")
        # Create a blank image of size 224x224
        blank_image = Image.new('RGB', (224, 224), (0, 0, 0))
        # Use the movie title as the file name, replacing spaces with underscores
        blank_file_name = f"{title.replace(' ', '_')}.jpg"
        blank_file_path = os.path.join(output_directory, blank_file_name)
        # Save the blank image
        blank_image.save(blank_file_path)
        print(f"Created blank image for: {title}")