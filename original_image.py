import pandas as pd
import numpy as np
from PIL import Image
import os

# image directories
ORIGINAL_FP_PREFIX: str = "/Users/gavintravis/Downloads/images_512/original/"
# ORIGINAL_FP_PREFIX: str = "/weka/scratch/project/hackathon/data/CropResiduePredictionChallenge/images_512/original/"
RESIDUE_FP_PREFIX: str = "/Users/gavintravis/Downloads/images_512/label/residue_background/"
# RESIDUE_FP_PREFIX: str = "/weka/scratch/project/hackathon/data/CropResiduePredictionChallenge/images_512/residue_background/"

# "/Users/gavintravis/Downloads/images_512/original/Zak-W-winterBarley_1m_20220401/IMG_0939.jpg_part03.jpg"

"""
    example filename by location type:
        {location}/{filename}.extension
        "Zak-W-winterBarley_1m_20220401/IMG_0939.jpg_part03.jpg"
        "Ritzville2-SprWheat1m20220329/IMG_0789_part04.jpg"
        "Ritzville3-WheatFallow1pass1m20220329/IMG_0807_part04.jpg"
        "Ritzville6-SprWheatWintPeas1m20220329/IMG_0895_part08.jpg"
        "Limbaugh1-1m20220328/IMG_0638.jpg_part08.jpg"
"""

class original_image:
    location: str
    image_num: int
    part_num: int
    data: pd.DataFrame

    def __init__(self, abs_filepath: str):
        self.file_path: str = abs_filepath[abs_filepath.find("original/", )+9:]
        self.location: int = -1
        self.image_num: int = -1
        self.part_num: int = -1

        self.location, self.image_num, self.part_num = self.parse_filename_by_location(self.file_path)

        # open and read original image file into pd data frame
        self.data = self.agAid_image_loader(abs_filepath)

        #add new columns with classification labels
        self.data["location"] = self.map_location(self.location)
        self.data["image_num"] = self.image_num
        self.data["part_num"] = self.part_num

        # search for matching residue image
        residue_df = self.get_residue_label()["red"]
        residue_df.rename("residue", inplace=True)
        if "Ritzville" in self.location:
            residue_df = residue_df.apply(lambda x: True if x == 255 else False)
        else:
            residue_df = residue_df.apply(lambda x: True if x == 0 else False)
        
        self.data = self.data.join(residue_df)

        print(self.data)

    def map_location(self, location: str) -> int:
        location_value: int = -1
        match location:
            case "Zak-W-winterBarley_1m_20220401":
                location_value = 0
            case "Ritzville2-SprWheat1m20220329":
                location_value = 1
            case "Ritzville3-WheatFallow1pass1m20220329":
                location_value = 2
            case "Ritzville6-SprWheatWintPeas1m20220329":
                location_value = 3
            case "Limbaugh1-1m20220328":
                location_value = 4
            case _:
                assert(False)

        return location_value
            


    
    def parse_filename_by_location(self, filepath: str) -> tuple[str, int, int]:
        file: list[str] = filepath.split("/")
        location: int = file[0]
        filename: str = file[1]
        start: int = filename.find("IMG_") + 4
        end: int = -1
        match location:
            case "Zak-W-winterBarley_1m_20220401" | "Limbaugh1-1m20220328":
                end = filename.find(".", start)
            case _ :
                end = filename.find("_", start)
            
        image_num = int(filename[start:end])
        start = filename.find("part", end) + 4
        end = filename.find(".", start)
        part_num = int(filename[start:end])

        return (location, image_num, part_num)
    
    def agAid_image_loader(self, filepath) -> pd.DataFrame:
        colourImg = Image.open(filepath)
        colourPixels = colourImg.convert("RGB")
        colourArray = np.array(colourPixels.getdata()).reshape(colourImg.size + (3,))
        indicesArray = np.moveaxis(np.indices(colourImg.size), 0, 2)
        allArray = np.dstack((indicesArray, colourArray)).reshape((-1, 5))


        return pd.DataFrame(allArray, columns=["y", "x", "red","green","blue"])
    
    # Needs the abs filepath for the kamiak file location
    def get_residue_label(self) -> pd.DataFrame:
        image_found: bool = False
        part_found: bool = False
        fp = ""
        for folder in os.listdir(fp := os.path.join(RESIDUE_FP_PREFIX, self.location + "/")):
            try:
                if int(folder.strip("IMG_")) == self.image_num:
                    fp = os.path.join(fp, folder + "/")
                    image_found = True
                    break
            except ValueError:
                continue
        if image_found:
                for image in list(filter(lambda x: x.endswith(".tif"), os.listdir(fp))):
                    try:
                        if int(image.strip(".tif")[-2:]) == self.part_num:
                            fp = os.path.join(fp, image)
                            part_found = True
                            break
                    except ValueError:
                        continue

        return self.agAid_image_loader(fp) if image_found and part_found else pd.DataFrame()

def test() -> bool:
    test =original_image("/Users/gavintravis/Downloads/images_512/original/Zak-W-winterBarley_1m_20220401/IMG_0939.jpg_part03.jpg")
    test = original_image("/Users/gavintravis/Downloads/images_512/original/Ritzville2-SprWheat1m20220329/IMG_0789_part04.jpg")
    test = original_image("/Users/gavintravis/Downloads/images_512/original/Ritzville3-WheatFallow1pass1m20220329/IMG_0807_part04.jpg")
    test = original_image("/Users/gavintravis/Downloads/images_512/original/Ritzville6-SprWheatWintPeas1m20220329/IMG_0895_part08.jpg")
    test = original_image("/Users/gavintravis/Downloads/images_512/original/Limbaugh1-1m20220328/IMG_0638.jpg_part08.jpg")

    return True

def run() -> None:
    root_dir: str = "/scratch/project/hackathon/data/CropResiduePredictionChallenge/images_512/original/"
    #root_dir: str = "/Users/gavintravis/Downloads/images_512/original/"
    file_count: int = 0

    for folder in os.listdir(root_dir):
            if folder == ".DS_Store": continue
            
            for image in os.listdir(fp := os.path.join(root_dir, folder + "/")):
                
                new_image_table = original_image(os.path.join(fp, image))
                new_image_table.data.to_csv("data.csv", mode="a", index=False)
                file_count += 1
                print(1)
                del new_image_table
