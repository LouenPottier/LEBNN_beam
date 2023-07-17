# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:08:35 2023

@author: LP263296
"""

import streamlit as st 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import torch
from streamlit_plotly_events import plotly_events

from load_models import fcnn, fclat, enerlat, enc_f, dec_f, H_fclat, H_enerlat



if 'u_fcnn' not in st.session_state:
    st.session_state['u_fcnn'] = fcnn.u0(1)

if 'h_fclat' not in st.session_state:
    st.session_state['h_fclat'] = fclat.u_lat_0(1)

if 'h_enerlat' not in st.session_state:
    st.session_state['h_enerlat'] = enerlat.u_lat_0(1)


with st.sidebar:
    st.subheader('Loading to apply at the free end of the beam')
    
    fx = st.slider('fx',-20000,20000,0,50)
    fy = st.slider('fy',-20000,20000,0,50) 
    #f=torch.tensor([[fx,fy]],dtype=torch.float32)/5000
    
    def applique(fx,fy):
        f_lat=torch.tensor([[fx,fy]],dtype=torch.float32)/5000
        f = dec_f(f_lat)
        
    
        st.session_state['u_fcnn'] = fcnn.adam(st.session_state['u_fcnn'],f,lr=0.1,nmax=500)
        
        st.session_state['h_fclat'] = fclat.adam_lat(st.session_state['h_fclat'],f_lat,lr=0.1,nmax=500)
        
        st.session_state['h_enerlat'] = enerlat.adam_lat(st.session_state['h_enerlat'],f_lat,lr=0.1,nmax=500)
        
        
    st.button('Apply load',on_click=applique,args=(fx,fy))
    
    
    
    st.subheader('Back to origin in latent space')

    
    def raz():
        st.session_state['u_fcnn'] = fcnn.u0(1)
        st.session_state['h_fclat'] = fclat.u_lat_0(1)
        st.session_state['h_enerlat'] = enerlat.u_lat_0(1)
        st.session_state['u_phy'] = fcnn.u0(1)
    
    st.button('Reset',on_click=raz)
    
    
    


##### PLOT FCNN

u=st.session_state['u_fcnn']
f_pred=fcnn(u)
fpx=f_pred[0,0].detach()*5000
fpy=f_pred[0,1].detach()*5000



fig1 = make_subplots(rows=1, cols=2)


#subplot1
fig1.add_trace(
    go.Scatter(x=u[0,:,0].detach(),y=u[0,:,1].detach()),
    row=1, col=1
)



fig1.add_annotation(x =  u[0,-1,0].detach()+fx/50000,
                   y =  u[0,-1,1].detach()+fy/50000,
                   text = "",
                   xref = "x",
                   yref = "y",
                   showarrow = True,
                   arrowcolor='red',
                   arrowhead = 3,
                   arrowsize = 2,
                   ax = u[0,-1,0].detach(),
                   ay = u[0,-1,1].detach(),
                   axref="x",
                   ayref='y',)


fig1.add_annotation(x =  u[0,-1,0].detach()+fpx/50000,
                   y =  u[0,-1,1].detach()+fpy/50000,
                   text = "",
                   xref = "x",
                   yref = "y",
                   showarrow = True,
                   arrowcolor='green',
                   arrowhead = 3,
                   arrowsize = 2,
                   ax = u[0,-1,0].detach(),
                   ay = u[0,-1,1].detach(),
                   axref="x",
                   ayref='y',)
     



fig1.update_layout(xaxis_range=[-1,1],yaxis_range=[-1,1])

#st.plotly_chart(fig1)




##### PLOT FCLAT

st.subheader("Neural Network without energy structure")

u=fclat.dec_u(st.session_state['h_fclat'])
f_pred=fclat(u)

fpx=f_pred[0,0].detach()*5000
fpy=f_pred[0,1].detach()*5000

fig2 = make_subplots(rows=1, cols=2,  subplot_titles=("Backward prediction", "Latent space visualization"))

#subplot1
fig2.add_trace(
    go.Scatter(x=u[0,:,0].detach(),y=u[0,:,1].detach(),     name='',
),
    row=1, col=1,

)



fig2.add_annotation(x =  u[0,-1,0].detach()+fx/50000,
                   y =  u[0,-1,1].detach()+fy/50000,
                   text = "",
                   xref = "x",
                   yref = "y",
                   showarrow = True,
                   arrowcolor='red',
                   arrowhead = 3,
                   arrowsize = 2,
                   ax = u[0,-1,0].detach(),
                   ay = u[0,-1,1].detach(),
                   axref="x",
                   ayref='y',) 

fig2.add_annotation(x =  u[0,-1,0].detach()+fpx/50000,
                   y =  u[0,-1,1].detach()+fpy/50000,
                   text = "",
                   xref = "x",
                   yref = "y",
                   showarrow = True,
                   arrowcolor='green',
                   arrowhead = 3,
                   arrowsize = 2,
                   ax = u[0,-1,0].detach(),
                   ay = u[0,-1,1].detach(),
                   axref="x",
                   ayref='y',)
     



fig2.update_layout(xaxis_range=[-1.1,1.1],yaxis_range=[-1.1,1.1])



#subplot2
n=200

x = torch.linspace(-4, 4, n)
y = torch.linspace(-2.5, 5.5, n)
X, Y = torch.meshgrid(x, y)

h=torch.cat((X.flatten().unsqueeze(1),Y.flatten().unsqueeze(1)),1)
h=torch.cat((X.flatten().unsqueeze(1),Y.flatten().unsqueeze(1)),1)
f=torch.tensor([[fx,fy]],dtype=torch.float32)/5000
n=len(x)
nf=f.expand(n*n,2)

nZ=( torch.sum((f-fclat.fcnn(h))**2,-1)).view(n,n).detach()
    



fig2.add_trace(
    go.Scatter(x=H_fclat[:,0].detach(),y=H_fclat[:,1].detach(), mode='markers', opacity=0.5,
    marker=dict(
                    color='LightSkyBlue',
                    size=4,
    ),
    name=''),
    row=1, col=2
)


fig2.add_trace(
    go.Scatter(x=st.session_state['h_fclat'][:,0].detach(),y=st.session_state['h_fclat'][:,1].detach(), mode='markers',  name=''), 
    row=1, col=2
)





fig2.add_trace(
   go.Contour(
           z=nZ.transpose(1,0),
           x=x, # horizontal axis
           y=y,
       colorscale='RdBu',
       line = dict(width = 0.1),
       contours=dict(
           start=0,
           end=20,
           size=0.5,
       ),
       colorbar=dict(title='||RN(h)-f||²', len=0.6),

   ),
    row=1, col=2
)


st.plotly_chart(fig2)




##### PLOT ENERLAT
st.subheader("Neural Network with energy structure (LEBNN)")


u=enerlat.dec_u(st.session_state['h_enerlat'])
f_pred=fclat(u)

fpx=f_pred[0,0].detach()*5000
fpy=f_pred[0,1].detach()*5000

fig3 = make_subplots(rows=1, cols=2, subplot_titles=("Backward prediction", "Latent space visualization"))

#subplot1
fig3.add_trace(
    go.Scatter(x=u[0,:,0].detach(),y=u[0,:,1].detach(),    name=''),
    row=1, col=1
)



fig3.add_annotation(x =  u[0,-1,0].detach()+fx/50000,
                   y =  u[0,-1,1].detach()+fy/50000,
                   text = "",
                   xref = "x",
                   yref = "y",
                   showarrow = True,
                   arrowcolor='red',
                   arrowhead = 3,
                   arrowsize = 2,
                   ax = u[0,-1,0].detach(),
                   ay = u[0,-1,1].detach(),
                   axref="x",
                   ayref='y',)


fig3.add_annotation(x =  u[0,-1,0].detach()+fpx/50000,
                   y =  u[0,-1,1].detach()+fpy/50000,
                   text = "",
                   xref = "x",
                   yref = "y",
                   showarrow = True,
                   arrowcolor='green',
                   arrowhead = 3,
                   arrowsize = 2,
                   ax = u[0,-1,0].detach(),
                   ay = u[0,-1,1].detach(),
                   axref="x",
                   ayref='y',)
     



fig3.update_layout(xaxis_range=[-1.1,1.1],yaxis_range=[-1.1,1.1])



#subplot2





n=200

x = torch.linspace(-1.5, 1.5, n)
y = torch.linspace(-1.5, 1.5, n)
X, Y = torch.meshgrid(x, y)

h=torch.cat((X.flatten().unsqueeze(1),Y.flatten().unsqueeze(1)),1)
h=torch.cat((X.flatten().unsqueeze(1),Y.flatten().unsqueeze(1)),1)
f=torch.tensor([[fx,fy]],dtype=torch.float32)/5000
n=len(x)
nf=f.expand(n*n,2)

nZ=(enerlat.V(h)-enerlat.W(h,nf)).view(n,n).detach()


fig3.add_trace(
    go.Scatter(x=H_enerlat[:,0].detach(),y=H_enerlat[:,1].detach(), mode='markers', opacity=0.5,
    marker=dict(
                    color='LightSkyBlue',
                    size=4,
    ),
    name=''),
    row=1, col=2
)


fig3.add_trace(
    go.Scatter(x=st.session_state['h_enerlat'][:,0].detach(),y=st.session_state['h_enerlat'][:,1].detach(), mode='markers',  name='',
), 
    row=1, col=2
)





fig3.add_trace(
   go.Contour(
           z=nZ.transpose(1,0),
           x=x, # horizontal axis
           y=y,
       colorscale='RdBu',
       line = dict(width = 0.1),
       contours=dict(
           start=-1,
           end=1,
           size=0.05,
       ),
       colorbar=dict(title='V(h) - h^t. B.f', len=0.6),
   ),
    row=1, col=2
)



st.plotly_chart(fig3)




