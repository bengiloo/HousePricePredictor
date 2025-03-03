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


def importHousing(path):
    columns = ["bed_count","bath_count","square_ft","zipcode","price"]
    df = pd.read_csv(path,names=columns,sep=" ")
    
    # print sample
    print(df.head())

    # inquire dataset variety
    print("\nBedrooms: ")
    print(df["bed_count"].value_counts())
    
    print("\nBathrooms: ")
    print(df["bath_count"].value_counts())

    print("\nSquare_ft")
    print("min: {}".format(df["square_ft"].min()))
    print("mean: {}".format(df["square_ft"].mean()))
    print("max: {}".format(df["square_ft"].max()))

    print("\nZipcodes: ")
    print(df["zipcode"].value_counts())

    print("\nPrice")
    print("min: {}".format(df["price"].min()))
    print("mean: {}".format(df["price"].mean()))
    print("max: {}".format(df["price"].max()))
    
    return df

# Main function
def main():
    path = ".\Houses_Dataset\HousesInfo.txt"
    housing_df = importHousing(path)




if __name__=="__main__":
    main()