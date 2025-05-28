# pages/second_page.py

import streamlit as st
st.set_page_config(page_title="Model visualisations", page_icon="📊")

train_losses = st.session_state["train_losses"] 
val_losses = st.session_state["val_losses"] 

import streamlit as st
import numpy as np

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.title("📊 Model training loss")

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots()
plt.plot(train_losses, label="Train Loss", marker='o', linestyle='-', color='#FF6F61', linewidth=2, markersize=5, )
plt.plot(val_losses, label="Validation Loss", marker='*', linestyle='-', color="#690F0F", linewidth=2, markersize=5, )
plt.xlabel("Epoch", fontsize=14, labelpad=10)
plt.ylabel("Loss", fontsize=14, labelpad=10)
plt.title("Loss per Epoch")
plt.legend()

fig.patch.set_facecolor('#f9f9f9')
ax.set_facecolor('#ffffff')
plt.tight_layout()
st.pyplot(fig)
