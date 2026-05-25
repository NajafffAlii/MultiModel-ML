import os
import random
import numpy as np
import cv2

def create_synthetic_dataset(base_dir="data_extracted/Houses-dataset-master/Houses Dataset", num_houses=200):
    os.makedirs(base_dir, exist_ok=True)
    
    # 1. Generate HousesInfo.txt
    info_path = os.path.join(base_dir, "HousesInfo.txt")
    zipcodes = [90210, 94109, 33139, 10001, 60611]
    
    with open(info_path, 'w') as f:
        for i in range(1, num_houses + 1):
            bedrooms = random.randint(1, 6)
            bathrooms = random.randint(1, 4)
            area = random.randint(800, 5000)
            zipcode = random.choice(zipcodes)
            
            # Price roughly depends on area, bedrooms, bathrooms and zipcode
            base_price = area * 150 + bedrooms * 10000 + bathrooms * 15000
            zip_multiplier = {90210: 2.5, 94109: 2.0, 33139: 1.5, 10001: 2.2, 60611: 1.2}[zipcode]
            price = int(base_price * zip_multiplier * random.uniform(0.9, 1.1))
            
            f.write(f"{bedrooms} {bathrooms} {area} {zipcode} {price}\n")
            
            # 2. Generate 4 synthetic images per house (frontal, bedroom, bathroom, kitchen)
            views = ["frontal", "bedroom", "bathroom", "kitchen"]
            colors = {
                "frontal": (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)),
                "bedroom": (random.randint(100, 255), random.randint(100, 255), 100),
                "bathroom": (100, random.randint(100, 255), random.randint(100, 255)),
                "kitchen": (random.randint(100, 255), 100, random.randint(100, 255))
            }
            
            for view in views:
                # Create a simple colored image with some noise
                img = np.zeros((32, 32, 3), dtype=np.uint8)
                img[:] = colors[view]
                
                # Add noise
                noise = np.random.randint(0, 50, (32, 32, 3), dtype=np.uint8)
                img = cv2.add(img, noise)
                
                # Save image
                img_path = os.path.join(base_dir, f"{i}_{view}.jpg")
                cv2.imwrite(img_path, img)
                
    print(f"Synthetic dataset with {num_houses} houses created successfully at {base_dir}")

if __name__ == "__main__":
    create_synthetic_dataset()
