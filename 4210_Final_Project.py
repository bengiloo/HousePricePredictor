# ===========================================================================================================
# Project: 4210 Housing Price Regression in neural network framework
# Contributors: Reyna Nava, Benjamin Luu, Miguelangel Soria
# Course: CS4210 "Machine Learning and Its Applications"
# ===========================================================================================================
import pandas as pd
from zipfile import ZipFile
import os
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import gc

# Data Importing and Pre-processing (Tabular Data)
# ===========================================================================================================
# extract the images
zip_path = "/content/Houses_Dataset.zip"
with ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("/content")

# read the tabular data
path = "/content/Houses_Dataset/HousesInfo.txt"
columns = ["bed_count","bath_count","square_ft","zipcode","price"]
df = pd.read_csv(path,names=columns,sep=" ")

# data type conversions
df["zipcode"] = df["zipcode"].astype(str)
df["bed_count"] = df["bed_count"].astype(int)
df["bath_count"] = df["bath_count"].astype(int)
df["square_ft"] = df["square_ft"].astype(int)
df["price"] = df["price"].astype(float)

# One-hot-encode zipcodes
encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
zipcodes_encoded = encoder.fit_transform(df[["zipcode"]])
zipcodes_encoded = pd.DataFrame(zipcodes_encoded, columns=encoder.get_feature_names_out(["zipcode"]))
labels = df[["price"]]
tabular_data = df.drop(['zipcode','price'], axis=1)
tabular_data = pd.concat([tabular_data, zipcodes_encoded], axis=1)

# Create the MinMaxScaler and normalize all numerical features
scaler = MinMaxScaler()
labels = scaler.fit_transform(labels.values.reshape(-1, 1))
tabular_data['square_ft'] = scaler.fit_transform(tabular_data['square_ft'].values.reshape(-1, 1))
tabular_data['bath_count'] = scaler.fit_transform(tabular_data['bath_count'].values.reshape(-1, 1))
tabular_data['bed_count'] = scaler.fit_transform(tabular_data['bed_count'].values.reshape(-1, 1))

# Data Importing and Pre-processing (Image Data)
# ===========================================================================================================
# load the names
image_names = os.listdir("/content/Houses_Dataset/")

# filter out .txt file
image_names = [name for name in image_names if name.endswith(".jpg")]

# sort based on the prefixed number AND sort by room
image_names = sorted(image_names,key=lambda x:(int(x.split("_")[0]), x.split("_")[1]))

#imgnet mean + std err
imgnet_mean = [0.485,0.456,0.406]
imgnet_std = [0.229,0.224,0.225]

# an array of the tensor images
transformed_images = []

# the image transformation function
transform = transforms.Compose([transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=imgnet_mean,std=imgnet_std)])

# transform each image
for image in image_names:
    img = Image.open("/content/Houses_Dataset/"+image)
    img_transformed = transform(img)
    transformed_images.append(img_transformed)

# convert to batch tensor
batch = torch.stack(transformed_images)

# clean up
del img_transformed
del imgnet_mean,imgnet_std
del transform
del img


# Prepare dataset
# ===========================================================================================================
# group images by house (4 PER HOUSE)
batch = batch.view(-1,4,3,224,224)

# size: 535 houses, 4 images each, 3 channels, 224 height, 224 width
# load the pretrained model

# use GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# use imagenet weights since small sampleset
pretrained_resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')

# drop the final layer (classification)
feature_extractor = nn.Sequential(*list(pretrained_resnet.children())[:-1])

# switch to evaulation mode to turnoff dropout + gradient
feature_extractor.eval()

# move to GPU
feature_extractor.to(device)

# configure batch and dataset
dataset = TensorDataset(batch)
dataloader = DataLoader(dataset, batch_size=2)

img_features = []

with torch.no_grad():
    # for each batch (two houses)
    for (images,) in dataloader:
      # move to cpu
      images = images.to(device)

      # get dims
      B,V,C,H,W = images.shape
      images = images.view(B*V,C,H,W)# (2*4,3,224,224)

      # get feats
      features_extracted = feature_extractor(images)
      # flatten
      features_extracted = features_extracted.view(B*V,-1)
      #reshape
      features_extracted = features_extracted.view(B,V,-1) #(2,4,512)
      # concat
      features_combined = features_extracted.view(B,-1) #(2,2048)

      # copy to cpu + allow gpu to clean it
      img_features.append(features_combined.cpu())

img_features = torch.cat(img_features,dim=0)
print(img_features.dtype)
print(img_features.shape) #(sample size, compiled images)

# clean memory
del pretrained_resnet
del feature_extractor
del images
del features_extracted
del features_combined
del batch

# Custom neural network
# ===========================================================================================================
class HousingPriceModel(torch.nn.Module):

    # constructor
    def __init__(self):
        super(HousingPriceModel,self).__init__()

        # input layer 1 - tabular data
        self.text_input = nn.Sequential(
            # cast 52 feat data to 128 -> activate -> 128
            nn.Linear(in_features=52,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=128),
        )

        # input layer 2 - image data
        # 2048 dim image (all 4 images composed into 1)
        self.img_input = nn.Sequential(
            nn.Linear(in_features=2048,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=128),
        )

        # dense layer fusing text and image
        self.fc = nn.Sequential(
            nn.Linear(in_features=256,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=64)
        )

        # dense layer
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=64,out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32,out_features=1)
        )


    # feed forward
    # outline steps for how the layers will connect
    def forward(self,x_imgs,x_text):

      # pass inputs + activate
      img_input = self.img_input(x_imgs)
      text_input = self.text_input(x_text)

      # combine
      inputs = torch.cat([img_input,text_input],dim=1)

      # hidden layers
      x = self.fc(inputs)
      y = self.fc2(x)
      return y.squeeze() # reduce to 1 dim

# primary training loop
# ===========================================================================================================

# configure kfold split
kf = KFold(n_splits=5,shuffle=False)

# to keep all the results
train_results = []
val_results = []
test_results = []

# to keep the best model
best_overall_loss = float('inf')
best_overall_model_state = None
best_overall_fold = None
best_preds = []
best_labels = []


for fold, (train_index, test_index) in enumerate(kf.split(img_features)):
  print(f"\nFold {fold + 1}=============================================")

  # split the dataset into train/test/val ===================================================================
  # split is 64%/16%/20%
  train_split = int(0.8 * len(train_index))

  # images
  x_img_train= img_features[train_index][:train_split]
  x_img_val = img_features[train_index][train_split:]
  x_img_test = img_features[test_index]

  # tabular
  x_tab_train = tabular_data.iloc[train_index][:train_split]
  x_tab_val = tabular_data.iloc[train_index][train_split:]
  x_tab_test = tabular_data.iloc[test_index]

  # labels
  y_train = labels[train_index][:train_split]
  y_val = labels[train_index][train_split:]
  y_test = labels[test_index]

  # =========================================================================================================
  # conversion
  x_tab_train = torch.tensor(x_tab_train.values, dtype=torch.float32)
  x_tab_val = torch.tensor(x_tab_val.values, dtype=torch.float32)
  x_tab_test = torch.tensor(x_tab_test.values, dtype=torch.float32)
  y_train = torch.tensor(y_train, dtype=torch.float32)
  y_val = torch.tensor(y_val, dtype=torch.float32)
  y_test = torch.tensor(y_test, dtype=torch.float32)

  # group into tuple
  train_data = TensorDataset(x_img_train,x_tab_train,y_train)
  val_data = TensorDataset(x_img_val,x_tab_val,y_val)
  test_data = TensorDataset(x_img_test, x_tab_test, y_test)

  # group by loads (64 houses at once)
  train_loader = DataLoader(train_data,batch_size=64,shuffle=True)
  val_loader = DataLoader(val_data,batch_size=64)
  test_loader = DataLoader(test_data,batch_size=64)

  # =========================================================================================================
  # initalization
  model = HousingPriceModel().to(device)
  loss_fn = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=3,
                                                         factor=0.5,cooldown=4,min_lr=1e-6)
  prev_lr = optimizer.param_groups[0]['lr']

  # =========================================================================================================
  # 30 rounds of iterating over each training BATCH
  for epoch in range(30):
    
    # keep epoch-wise information
    model.train()
    train_loss = 0.0
    train_preds = []
    train_labels = []
    
    # for each epoch
    for (img_batch,tab_batch,label_batch) in train_loader:
      
      # move it all to GPU
      img_batch = img_batch.to(device)
      tab_batch = tab_batch.to(device)
      label_batch = label_batch.squeeze().to(device)

      # forward pass
      output = model(img_batch,tab_batch)

      # clear gradients of previous EPOCH
      optimizer.zero_grad()

      # compute MSE
      loss = loss_fn(output,label_batch)

      # backprop.
      loss.backward()
      optimizer.step()

      # sum the loss
      train_loss += loss.item()

      # metrics
      train_preds.extend(output.detach().cpu().numpy())
      train_labels.extend(label_batch.cpu().numpy())
  # =========================================================================================================
  # training metrics
    avg_train_loss = train_loss / len(train_loader)
    train_mse = mean_squared_error(train_labels, train_preds)
    train_mae = mean_absolute_error(train_labels, train_preds)
    train_r2 = r2_score(train_labels, train_preds)
    train_results.append((avg_train_loss,train_mse, train_mae,train_r2))
  # ========================================================================================================
  # after each epoch (1 view of all the data)
    model.eval()

    # keep track of metrics
    val_loss = 0.0
    all_preds = []
    all_labels = []

    # look at the validation subset
    with torch.no_grad():
      for (img_batch,tab_batch,label_batch) in val_loader:
        # move it all to GPU
        img_batch = img_batch.to(device)
        tab_batch = tab_batch.to(device)
        label_batch = label_batch.squeeze().to(device)

        # get outputs and compute loss
        output = model(img_batch, tab_batch).squeeze()
        loss = loss_fn(output,label_batch)
        val_loss += loss.item()

        # collect predictions/labels
        all_preds.extend(output.cpu().numpy())
        all_labels.extend(label_batch.cpu().numpy())

    #========================================================================================================
    # End of round
    # print train loss/validation loss
    avg_val_loss = val_loss / len(val_loader)
    val_mse = mean_squared_error(all_labels,all_preds)
    val_r2 = r2_score(all_labels, all_preds)
    val_mae = mean_absolute_error(all_labels, all_preds)
    val_results.append((avg_val_loss,val_mse, val_mae,val_r2))
    print(f"Epoch {epoch+1} | Train Loss: {train_loss / len(train_loader):.4f} | Train MSE: {train_mse:.4f} | MAE: {train_mae:.4f} | R²: {train_r2:.4f} | Val Loss: {avg_val_loss:.4f} | Val MSE: {val_mse:.4f} | MAE: {val_mae:.4f} | R²: {val_r2:.4f}")

    # MAKE A STEP IN LR
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    if current_lr < prev_lr:
      print(f"Epoch {epoch+1}: Learning rate reduced from {prev_lr:.6f} to {current_lr:.6f}")
    prev_lr = current_lr

    # =======================================================================================================
    
  #=========================================================================================================
    # add to list
    test_loss = 0.0
    test_preds = []
    test_true_labels = []

    with torch.no_grad():
        for img_batch, tab_batch, label_batch in test_loader:
            img_batch, tab_batch, label_batch = img_batch.to(device), tab_batch.to(device), label_batch.squeeze().to(device)

            output = model(img_batch, tab_batch)
            loss = loss_fn(output, label_batch)
            test_loss += loss.item()

            test_preds.extend(output.cpu().numpy())
            test_true_labels.extend(label_batch.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    test_mse = mean_squared_error(test_true_labels, test_preds)
    test_r2 = r2_score(test_true_labels, test_preds)
    test_mae = mean_absolute_error(test_true_labels, test_preds)
    test_results.append((avg_test_loss,test_mse, test_mae,test_r2))

    # check if best model
    if avg_test_loss < best_overall_loss:
      best_overall_loss = avg_test_loss
      best_overall_model_state = model.state_dict()
      best_overall_fold = fold
      best_preds = all_preds
      best_labels = all_labels
  # =========================================================================================================

# house cleaning (of the GPU/RAM)
del model, optimizer, train_loader, val_loader, test_loader, train_data, val_data, img_batch, tab_batch, label_batch, output, loss
torch.cuda.empty_cache()
gc.collect()

# ======== Summary =========
avg_test_loss = sum(x[0] for x in test_results) / len(test_results)
test_mse = sum(x[1] for x in test_results) / len(test_results)
test_mae = sum(x[2] for x in test_results) / len(test_results)
test_r2 = sum(x[3] for x in test_results) / len(test_results)

print("=================================================================")
print("\nTest Results:")
for i in range(5):
  print(f"Fold {i+1} | Test Loss: {avg_test_loss:.4f} | MSE: {test_results[i][0]:.4f} | MAE: {test_results[i][1]:.4f} | R²: {test_results[i][2]:.4f}")

print(f"\nAverage Test MSE: {test_mse:.4f}")
print(f"Average Test MAE: {test_mae:.4f}")
print(f"Average Test R²: {test_r2:.4f}")

# Save best model after all folds
if best_overall_model_state is not None:
    torch.save(best_overall_model_state, f"Housing Price Predictor{best_overall_fold}.pth")
    print(f"\n Saved best model from Fold {best_overall_fold} with MSE: {best_overall_loss:.4f}")

# exporting of results 
# ===========================================================================================================
train_cols = ["Train Loss","Train MSE", "Train MAE", "Train R2"]

train_fold1_results = pd.DataFrame(train_results[:30], columns=train_cols)
train_fold2_results = pd.DataFrame(train_results[30:60], columns=train_cols)
train_fold3_results = pd.DataFrame(train_results[60:90], columns=train_cols)
train_fold4_results = pd.DataFrame(train_results[90:120], columns=train_cols)
train_fold5_results = pd.DataFrame(train_results[120:150], columns=train_cols)

v_cols = ["Val Loss","Val MSE", "Val MAE", "Val R2"]

val_fold1_results = pd.DataFrame(val_results[:30], columns=v_cols)
val_fold2_results = pd.DataFrame(val_results[30:60], columns=v_cols)
val_fold3_results = pd.DataFrame(val_results[60:90], columns=v_cols)
val_fold4_results = pd.DataFrame(val_results[90:120], columns=v_cols)
val_fold5_results = pd.DataFrame(val_results[120:150], columns=v_cols)

tst_cols = ["Test Loss","Test MSE", "Test MAE", "Test R2"]
test_fold1_results = pd.DataFrame(test_results[:30], columns=tst_cols)
test_fold2_results = pd.DataFrame(test_results[30:60], columns=tst_cols)
test_fold3_results = pd.DataFrame(test_results[60:90], columns=tst_cols)
test_fold4_results = pd.DataFrame(test_results[90:120], columns=tst_cols)
test_fold5_results = pd.DataFrame(test_results[120:150], columns=tst_cols)
