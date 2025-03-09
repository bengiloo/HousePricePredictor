# ===================================================================================
# Project: 4210 Final Project - Housing Estimation
# Programmers: Reyna Nava, Benjamin Luu, Miguelangel Soria M.
# Due Date: 05/xx/25
# Description: 
#   ---
# ===================================================================================

# Libraries
import numpy as np
import pandas as pd
import cv2
import os
import statistics

# = zip_encoded =====================================================================







# = import_house_info ===============================================================
# Purpose:
#   Reads a textfile containing information about the housing dataset
# Parameters:
#   path: File path of the housing textfile
# Returns: 
#   df: a pandas dataframe containing the housing information
#   rows: number of rows in the dataframe
# ===================================================================================
def import_house_info(path):

    # Column Names
    columns = ["bed_count","bath_count","square_ft","zipcode","price"]
    df = pd.read_csv(path,names=columns,sep=" ")
    rows = len(df)

    # inquire dataset variety
    print("\nBedrooms: ")
    print(df["bed_count"].value_counts())
    
    print("\nBathrooms: ")
    print(df["bath_count"].value_counts())

    print("\nSquare ft: ")
    print("min: {}".format(df["square_ft"].min()))
    print("mean: {}".format(df["square_ft"].mean()))
    print("max: {}".format(df["square_ft"].max()))

    print("\nZipcodes: ")
    print(df["zipcode"].value_counts())
    print("Unique Count: ", len(df["zipcode"].value_counts()))


    zip_median_price = df.groupby("zipcode")["price"].median()
    df["zip_median_price"] = df["zipcode"].map(zip_median_price)


    print("\nPrice: ")
    print("min: {}".format(df["price"].min()))
    print("mean: {}".format(df["price"].mean()))
    print("max: {}".format(df["price"].max()))

    # print sample
    print(df.head())
    
    # Return dataframe + size
    return df, rows

# = transform_images ================================================================
# Purpose:
#   ----
# Parameters:
#   src_path: File path of the original images
#   dest_path: File path of the transformed images
#   img_names: list of image names
# Returns: 
#   N/A
# ===================================================================================
def transform_images(src_path, dest_path, image_names):

    # average image size
    height, width = 590,850

    # Read each image as grayscale; resize; store
    for i in range(len(image_names)):
        img = cv2.imread(src_path + image_names[i],cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(width,height))
        cv2.imwrite(dest_path + image_names[i],img)

# = convert_image_data ==============================================================
# Purpose:
#   ----
# Parameters:
#   path: File path of images to read
# Returns: 
#   -----
# ===================================================================================
def convert_image_data(path, image_names):

    image_data = []

    for i in range(len(image_names)):
        img = cv2.imread(path+image_names[i])
        image_data.append = np.asarray(img)

    return image_data

# ===================================================================================
def main():

    # Define file paths
    textfile_path = "./Houses_Dataset/HousesInfo.txt"
    img_src_path = "./Houses_Dataset/"
    img_dest_path = "../Houses_Dataset_Cln/"
    
    # read info into dataframe
    housing_df,size = import_house_info(textfile_path)

    # get list of image names
    #image_names = os.listdir(img_src_path)

    # read + transform + store images
    #transform_images(img_src_path,img_dest_path,image_names[:-1])

    # get pixel data
    #image_data = convert_image_data(img_dest_path)





if __name__=="__main__":
    main()