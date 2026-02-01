import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from dotenv import load_dotenv
load_dotenv()
import os
data_path=os.getenv("DATA_PATH")
df=pd.read_csv(data_path)
df=df.fillna(0)
print(df.tail()) ### 23.698 tane veri var.
###Total Area , rooms , floors_total , living_area , last_price değerlerini csv dosyadan okuyup listeye kaydetme
total_area=df["total_area"]
total_area_list=[]
for i in total_area:
    total_area_list.append(i)
### rooms
rooms=df["rooms"]
rooms_list=[]
for x in rooms:
    rooms_list.append(x)
### floors_total
floors_total=df["floors_total"]
floors_total_list=[]
for q in floors_total:
    floors_total_list.append(q)
#living_area
living_area=df["living_area"]
living_area_list=[]
for a in living_area:
    living_area_list.append(a)
### last_price
last_price=df["last_price"]
last_price_list=[]
for e in last_price:
    last_price_list.append(e)

### indeks tanımlama
indeks=torch.randperm(23698)
train_indeks=indeks[0:14219]
validition_indeks=indeks[14219:20000]
test_indeks=indeks[20000:23698]

### tensor_tanımlama
tensor_specialist=torch.tensor(np.column_stack([total_area_list,rooms_list,floors_total_list,living_area_list]),dtype=torch.float32)
tensor_last_price=torch.tensor(np.column_stack([last_price_list]),dtype=torch.float32)


### tensor specialist ayırma işlemleri train ,validition,test olarak
tensor_one=tensor_specialist[train_indeks]
tensor_two=tensor_specialist[validition_indeks]
tensor_theree=tensor_specialist[test_indeks]

### tensor last price ayırma işlemleri train,validition,test olarak
tensor_price_one=tensor_last_price[train_indeks]
tensor_price_two=tensor_last_price[validition_indeks]
tensor_price_theree=tensor_last_price[test_indeks]


### normalizasyon fonksiyonu tanımlama
def max_min_normalizition(tensor,maks,minx):
    return (tensor-minx)/(maks-minx)

### tensor specialist 3 farklı kategoriye bölündü ,ardından train tensor kullanarak max ve min değerlerini bularak diğer tensorlere normalizasyon işlemi yapacağım
### max ve min bulma (bir)
tensor_one_maks=tensor_one.max(dim=0).values
tensor_one_minx=tensor_one.min(dim=0).values

### max ve min bulma (iki)
tensor_price_one_maks=tensor_price_one.max(dim=0).values
tensor_price_one_minx=tensor_price_one.min(dim=0).values


### normalizasyon işlemleri (bir)
tensor_train_norm=max_min_normalizition(tensor_one,tensor_one_maks,tensor_one_minx)
tensor_validition_norm=max_min_normalizition(tensor_two,tensor_one_maks,tensor_one_minx)
tensor_test_norm=max_min_normalizition(tensor_theree,tensor_one_maks,tensor_one_minx)


### normalizasyon işlemi (iki)
price_train_norm=max_min_normalizition(tensor_price_one,tensor_price_one_maks,tensor_price_one_minx)
price_validition_norm=max_min_normalizition(tensor_price_two,tensor_price_one_maks,tensor_price_one_minx)
price_test_norm=max_min_normalizition(tensor_price_theree,tensor_price_one_maks,tensor_price_one_minx)

### Verileri gruplanıdırıp rastgele karıştırma işlemi
train_dataset=TensorDataset(tensor_train_norm,price_train_norm)
train_loader=DataLoader(train_dataset,shuffle=True,batch_size=10000)

class HomePricePrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.shell1=nn.Linear(4,50)
        self.shell2=nn.Linear(50,1)
        self.relu=nn.ReLU()
    def forward(self,x):
        x=self.shell1(x)
        x=self.relu(x)
        x=self.shell2(x)
        return x
model=HomePricePrediction()
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.1)
for epoch in range(1000):
    for x,y in train_loader:
        prediction=model(x)
        loss=criterion(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch%100==0:
        print(f"The loss value is {loss.item()}")
model.eval()
with torch.no_grad():
    prediction=model(tensor_test_norm)
    loss=criterion(prediction,price_test_norm)
    print(f"The loss value is {loss.item()}")
print("Eğitim işlemi tamamdır!")
data=torch.tensor([[85,4,7,35]],dtype=torch.float32)
data=max_min_normalizition(data,tensor_one_maks,tensor_one_minx)
result=model.forward(data)
result=result*(tensor_price_one_maks - tensor_price_one_minx)+tensor_price_one_minx
print(f"The result is {result.item()}")

check_point={'model_state': model.state_dict(),
    'norm_maks': tensor_one_maks,
    'norm_min': tensor_one_minx,
    'price_maks': tensor_price_one_maks,
    'price_min': tensor_price_one_minx}

torch.save(check_point,"ev_fiyat_tahmin.pth")
















































































































































