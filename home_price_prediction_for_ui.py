### python -m streamlit run home_price_prediction_for_ui.py
import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from model import HomePricePrediction
checkpoint=torch.load("ev_fiyat_tahmin.pth")
model=HomePricePrediction()
model.load_state_dict(checkpoint['model_state'])
model.eval()
norm_maks = checkpoint['norm_maks']
norm_min = checkpoint['norm_min']
price_maks = checkpoint['price_maks']
price_min = checkpoint['price_min']
st.warning("Pay Attention to this!")
check_box1=st.checkbox("This website is not formal for pricer or buyer.This website was created for only education.Do you accept that ?",help="You have to click the button!")
if check_box1:
    st.header("HOME PRİCE PREDİCTİON")
    st.info("I used dataset about price of home so I added this dataset on the end of website!")
    total_area=st.slider("Total Area:",min_value=30,max_value=160,step=1,help="You can slide the function")
    rooms=st.select_slider("Rooms:",options=[1,2,3,4,5],help="You can select how many rooms do you want to prefer")
    floors_total=st.slider("Total Floors",min_value=1,max_value=15,help="You can slide the function")
    living_area_tuhaf=st.slider("Living Room Area",min_value=20,max_value=70,help="You can slide the function")
    if total_area and rooms and floors_total and living_area_tuhaf:
        tensor = torch.tensor(np.column_stack([total_area, rooms, floors_total, living_area_tuhaf]),dtype=torch.float32)
        tensor = (tensor - norm_min) / (norm_maks - norm_min)
        result = model.forward(tensor)
        result = result * (price_maks - price_min) + price_min
        st.info(f"Our prediction is {result.item()}")
        st.success("Thank you for using to me :)")