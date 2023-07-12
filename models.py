import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


from rbf import RBF, gaussian, multiquadric, bump2





def flatten(u):
    """
    flatten the displacement and remove the fixed node
    supports batched displacements
    """
    return u[...,1:,:].flatten(start_dim=-2) 


def unflatten(u_flat):
    """
    opposite of flatten(u)
    supports batched displacements
    """
    u_flat=F.pad(u_flat,(2,0))
    new_shape=list(u_flat.shape)[:-1]+[u_flat.size(-1)//2,2]
    return u_flat.reshape(new_shape)
    



class EncoderU(nn.Module):
    """
    Fully-connected neural network that encode the displacement field
    in a low dimmentional latent space
    """    
    def __init__(self,num_nodes,latent_size,hidden_size):
        super(EncoderU, self).__init__()
        self.enc=nn.Sequential(nn.Linear((num_nodes-1)*2, hidden_size),
                                     nn.ELU(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.ELU(),
                                     nn.Linear(hidden_size,  latent_size))
        self.num_nodes=num_nodes
    def forward(self, u):
        #flatten the displacement and remove the fixed node
        u=u[...,1:,:].flatten(start_dim=-2) 
        u0=self.u0(1)[...,1:,:].flatten(start_dim=-2) 
        u_lat=self.enc(u)#-self.enc(u0)
        return u_lat
    
    def u0(self,batch_size=None):        
        device=next(self.parameters()).device
        if batch_size is None:
            u0=torch.zeros(self.num_nodes,2,device=device)
        else:
            u0=torch.zeros(batch_size,self.num_nodes,2,device=device)
        u0[...,:,0]=torch.linspace(0,1,10,device=device)
        return u0  
    

class DecoderU(nn.Module):
    """
    Fully-connected neural network that decode the displacement low 
    dimmentionnal latent variable into the displacement space
    """    
    def __init__(self,num_nodes,latent_size,hidden_size):
        super(DecoderU, self).__init__()
        self.dec=nn.Sequential(nn.Linear(latent_size, hidden_size), 
                             nn.ELU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ELU(),
                             nn.Linear(hidden_size, (num_nodes-1)*2)) 
        
    def forward(self, u_lat):
        
        #decode and add the fixed node displacement
        u_flat=F.pad(self.dec(u_lat),(2,0)) 
        
        #supports batched and non-batched
        new_shape=list(u_flat.shape)[:-1]+[u_flat.size(-1)//2,2]
        
        #unflatten the displacement field
        u=u_flat.view(new_shape)
        return u


class EncoderF(nn.Module):
    """
    Trivial encoder for the loading field : keep only the non-zero components
    and unflatten
    """    
    def __init__(self,nonzero_nodes_index=[9]):
        super(EncoderF, self).__init__()
        self.nonzero_nodes_index=nonzero_nodes_index
        
    def forward(self, f):
        f_lat=f[...,self.nonzero_nodes_index,:].flatten(start_dim=-2) 
        return f_lat


class DecoderF(nn.Module):
    """
    Trivial decoder for the loading field : add zeros and unflatten
    """    
    def __init__(self,nonzero_nodes_index=[9],num_nodes=10):
        super(DecoderF, self).__init__()
        self.nonzero_nodes_index=nonzero_nodes_index
        self.num_nodes=num_nodes
        
    def forward(self, f_lat):

        #supports batched and non-batched
        old_shape=list(f_lat.shape)[:-1]+[f_lat.size(-1)//2,2]
        new_shape=list(f_lat.shape)[:-1]+[self.num_nodes,2]

        f=torch.zeros(new_shape, device=f_lat.device)
        
        #unflatten the displacement field
        f[...,self.nonzero_nodes_index,:]=f_lat.view(old_shape)      
        return f

'''
class V(torch.nn.Module):
    """
    Learnable potential energy using 
    """
    def __init__(self, latent_size, num_basis_function):
        super(V, self).__init__()
        self.a = Parameter(torch.randn(num_basis_function)/num_basis_function)
        self.s = Parameter(torch.randn(num_basis_function))
        self.c = Parameter(torch.randn(num_basis_function,latent_size))
        self.n=num_basis_function
        self.latent_size=latent_size
        
    def forward(self, u):
        """
        V(u) = sum_i a_i * sqrt(1+s_i*sum_j((u_j-c_ij)**2))
        """
        batch_size=u.size(0)
        R=u.expand(self.n,batch_size,self.latent_size).transpose(1,0)-self.c.expand(batch_size,self.n,self.latent_size)
        r=torch.sum(R**2,-1)  #size(batch_size, n)
        E=torch.einsum('n,bn->b',F.tanh(self.a), torch.sqrt(1+torch.einsum('bn,n->bn',r,torch.abs(self.s)))  ) + 1/2*torch.sum(u**2,-1)
        return E.unsqueeze(-1)
    
    def grad(self, u):
        """
         grad V = sum_i  a_i * s_i  sum_j (u_j-c_ij)/(1+s_i*sum((u_k-c_ik)**2))^(1/2)
        """
        batch_size=u.size(0)
        R=u.expand(self.n,batch_size,self.latent_size).transpose(1,0)-self.c.expand(batch_size,self.n,self.latent_size)
        r=torch.sum(R**2,-1)  #size(batch_size, n)
        sqrt=torch.sqrt(1+torch.einsum('bn,n->bn',r,torch.abs(self.s)))
        return  torch.einsum('n,bnh,bn->bh',F.tanh(self.a)*torch.abs(self.s),R,1/sqrt) + u
    
    def grad_(self,u):
        return torch.vmap(torch.func.jacrev(self.forward))(u)[:, 0, 0, :]
    
    def H(self,u):
        h=torch.vmap(torch.func.hessian(self.forward))(u)[:, 0, 0, :]
        return h
'''


class V(torch.nn.Module):
    """
    Learnable potential energy using radial basis neural network

    """

    def __init__(self, latent_size, num_basis_function):
        super(V, self).__init__()
        self.rbf = RBF(latent_size, num_basis_function, multiquadric)
        self.lin1 = torch.nn.Linear( num_basis_function, 1, bias=True)
        self.num_basis_function = num_basis_function
        self.latent_size = latent_size
    def forward(self, h):
        return F.tanh(self.lin1(self.rbf(h))) + 1/2*torch.sum(h**2,-1).unsqueeze(-1)
    def grad(self,h):
        gradV = torch.vmap(torch.func.jacrev(self.forward))(h)[:,0,0,:]
        return gradV
    def grad_(self,h):
        a=self.lin1.weight.data[0]
        s=torch.exp(self.rbf.log_sigmas.data)
        c=self.rbf.centres.data
    
        batch_size=h.size(0)
        R=h.expand(self.num_basis_function,batch_size,self.latent_size).transpose(1,0)-c.expand(batch_size,self.num_basis_function,self.latent_size)
        r=torch.sum(R**2,-1)  #size(batch_size, n)
        sqrt=torch.sqrt(1+torch.einsum('bn,n->bn',r,s))
        grad_rbf=torch.einsum('n,bnh,bn->bh',a*s,R,1/sqrt)
        rbf=self.lin1(self.rbf(h))
        #return  grad_rbf * (1-F.tanh(rbf)**2) + h
        #return  grad_rbf * (torch.exp(rbf).clip(0,1)) + h
        #return  grad_rbf  + h
        return  grad_rbf * (1-F.tanh(rbf)**2) + h

  
    def H(self,h):
        h=torch.vmap(torch.func.hessian(self.forward))(h)[:, 0,0,:,:]
        return h

    

class W(nn.Module):
    """
    Learnable bilinear work function 
    
    To speed-up inference time, we parametrize the bilinear transformation
    by its inverse matrix    
    """

    def __init__(self, latent_size):
        super(W, self).__init__()
        self.Binv = Parameter(torch.randn(latent_size, latent_size) / math.sqrt(latent_size))
        
    def forward(self, u,f):
        B=torch.linalg.inv(self.Binv)
        return torch.einsum('bi,ij,bj->b',u,B,f).unsqueeze(-1)
    
    def w(self,u):
        B=torch.linalg.inv(self.Binv)
        return torch.einsum('bi,ij->bj',u,B)
    
    def grad(self,u,f):
        B=torch.linalg.inv(self.Binv)
        return torch.einsum('ij,bj->bi',B,f)
    
    def gradw(self,u):
        B=torch.linalg.inv(self.Binv)
        return B.unsqueeze(0).expand(u.size(0),u.size(1),u.size(1))
    
    def invgradw(self,u):
        Binv = self.Binv.unsqueeze(0).expand(u.size(0),u.size(1),u.size(1))
        return Binv
    def H(self,u,f):
        return torch.zeros(u.size(0),u.size(1),u.size(1))





class FCNN(nn.Module):
    """
    Fully-connected neural network 
        f=dec(fcnn(u)) with dec the trivial loading decoder
    """    
    def __init__(self, hidden_size, nonzero_nodes_index=[9],num_nodes=10):
        super(FCNN, self).__init__()
        self.fcnn=nn.Sequential(nn.Linear((num_nodes-1)*2, hidden_size), 
                             nn.ELU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ELU(),
                             nn.Linear(hidden_size, len(nonzero_nodes_index)*2)
                             )
        
        self.enc_f=EncoderF(nonzero_nodes_index)
        self.dec_f=DecoderF(nonzero_nodes_index,num_nodes)
        
        self.num_nodes=num_nodes
        
    def forward(self, u):
        u_flat=flatten(u)
        f_lat = self.fcnn(u_flat)
        f=self.dec_f(f_lat)
        return f
    
    def u0(self,batch_size=None):
        device=next(self.parameters()).device
        if batch_size is None:
            u0=torch.zeros(self.num_nodes,2,device=device)
        else:
            u0=torch.zeros(batch_size,self.num_nodes,2,device=device)
        u0[...,:,0]=torch.linspace(0,1,10,device=device)
        return u0  
    
    def adam(self,u0,f,lr=0.01,nmax=100):
        """
        Solve the backward problem by gradient descent
        """
        u=nn.Parameter(u0)
        optimizer=torch.optim.Adam([u],lr=lr)
        
        for i in range(nmax):
            optimizer.zero_grad()
            err=torch.sum((f-self(u))**2)
            err.backward()
            optimizer.step()
        return u
    
    def backward(self,f,u0=None):        
        if f.dim()==2:
            batch_size=None
        else:
            batch_size=f.size(0)
        if u0 is None:
            u0=self.u0(batch_size)
        return self.adam(u0,f)






class FCLatNN(nn.Module):
    """
    Fully-connected neural network using a latent space for the displacements
    forward:    f=dec2(fcnn(enc1(u))) 
                                with dec2 the trivial loading encoder
                                and enc1 a fully-connected encoder
                                associated with a fully-connected decoder dec1                
    """    
    def __init__(self, latent_size, hidden_size, nonzero_nodes_index=[9],num_nodes=10):
        super(FCLatNN, self).__init__()
        
        self.fcnn=nn.Sequential(nn.Linear(latent_size, hidden_size), 
                             nn.ELU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ELU(),
                             nn.Linear(hidden_size, len(nonzero_nodes_index)*2)
                             )
        
        
        self.enc_u=EncoderU(num_nodes, latent_size, hidden_size)
        self.dec_u=DecoderU(num_nodes, latent_size, hidden_size)
        self.enc_f=EncoderF(nonzero_nodes_index)
        self.dec_f=DecoderF(nonzero_nodes_index,num_nodes)   
        
        self.num_nodes=num_nodes
        self.latent_size = latent_size
        
    def forward(self, u):
        u_lat=self.enc_u(u)
        f_lat=self.fcnn(u_lat)
        f=self.dec_f(f_lat)
        return f
    
    def autoenc(self, u):
        return self.dec_u(self.enc_u(u))
    
    def u0(self,batch_size=None):        
        device=next(self.parameters()).device
        if batch_size is None:
            u0=torch.zeros(self.num_nodes,2,device=device)
        else:
            u0=torch.zeros(batch_size,self.num_nodes,2,device=device)
        u0[...,:,0]=torch.linspace(0,1,10,device=device)
        return u0  
    
    def u_lat_0(self,batch_size=None):
        return self.enc_u(self.u0(batch_size))
           
    def adam(self,u0,f,lr=0.01,nmax=100):
        """
        Solve the backward problem by gradient descent in the displacement space
        """
        u=nn.Parameter(u0)
        optimizer=torch.optim.Adam([u],lr=lr)

        for i in range(nmax):
            optimizer.zero_grad()
            err=torch.sum((f-self(u))**2)
            err.backward()
            optimizer.step()
        return u.detach()
    
    def adam_lat(self,u_lat_0,f_lat,lr=0.01,nmax=100):
        """
        Solve the backward problem by gradient descent in the latent space
        """
        u_lat=nn.Parameter(u_lat_0)
        optimizer=torch.optim.Adam([u_lat],lr=lr)
        for i in range(nmax):
            optimizer.zero_grad()
            err=torch.sum((f_lat-self.fcnn(u_lat))**2)
            err.backward()
            optimizer.step()
        return u_lat.detach()
    
    def backward(self,f,u0=None):
        if f.dim()==2:
            batch_size=None
        else:
            batch_size=f.size(0)
        if u0 is None:
            u_lat_0=self.u_lat_0(batch_size)
        else:
            u_lat_0=self.enc_u(u0)
        f_lat=self.enc_f(f)
        u=self.dec_u(self.adam_lat(u_lat_0,f_lat))
        return u





class EnergyLatNN(nn.Module):
    """  
    Neural network using a latent space in which a learnable energy is
    conserved to learn the displacemets - loading relation.
    
    The conserved energy is V(u_lat)-W(u_lat,f_lat)
    V is a radially unbounded function
    W is a bilinear function : W(u_lat, f_lat) = u_lat^t .B.f_lat
    
    f = dec2( B^-1 .grad(V(enc1(u))) )
    
    non-batched inputs are not supported
    """    
    def __init__(self,latent_size, hidden_size, radial_basis_functions, nonzero_nodes_index=[9],num_nodes=10):
        super(EnergyLatNN, self).__init__()
        
        self.enc_u=EncoderU(num_nodes, latent_size, hidden_size)
        self.dec_u=DecoderU(num_nodes, latent_size, hidden_size)
        self.enc_f=EncoderF(nonzero_nodes_index)
        self.dec_f=DecoderF(nonzero_nodes_index,num_nodes)   
        
        
        self.V=V(latent_size,radial_basis_functions) 
        self.W=W(latent_size)
        
        self.num_nodes = num_nodes
    def forward(self,u):        

        u_lat=self.enc_u(u)
        
        gradV=self.V.grad(u_lat)
        
        invgradw=self.W.Binv #self.W.invgradw(u_lat)
        
        f_lat=torch.einsum('ik,...k->...i',invgradw,gradV)
        f=self.dec_f(f_lat)
        return f
    
    def E(self,u_lat,f_lat):
        return self.V(u_lat)-self.W(u_lat,f_lat)
    
    def E_(self,u,f):
        u_lat=self.enc_u(u)
        f_lat=self.enc_f(f)
        return self.E(u_lat,f_lat)
    
    def autoenc(self, u):
        return self.dec_u(self.enc_u(u))
    
    def u0(self,batch_size=1):        
        device=next(self.parameters()).device
        if batch_size is None:
            u0=torch.zeros(self.num_nodes,2,device=device)
        else:
            u0=torch.zeros(batch_size,self.num_nodes,2,device=device)
        u0[...,:,0]=torch.linspace(0,1,10,device=device)
        return u0  
    
    def u_lat_0(self,batch_size=1):
        return self.enc_u(self.u0(batch_size))
    
    def newton(self,u0,f,nmax=100,lr=1.,tol=1.e-5,plot=False):
        u_lat=self.enc_u(u0)
        f_lat=self.enc_f(f)
        
        i=0
        while torch.sum((self.V.grad(u_lat)-self.W.grad(u_lat,f_lat))**2)>tol and i<nmax:
            i+=1
            u_lat=u_lat-torch.linalg.solve(self.V.H(u_lat),self.V.grad(u_lat)-self.W.grad(u_lat,f_lat))
        return self.dec_u(u_lat)

    def newton_lat(self,u_lat_0,f_lat,nmax=100,lr=1.,tol=1.e-5,plot=False):
        u_lat=u_lat_0
        
        i=0
        while torch.sum((self.V.grad(u_lat)-self.W.grad(u_lat,f_lat))**2)>tol and i<nmax:
            i+=1
            u_lat=u_lat-torch.linalg.solve(self.V.H(u_lat),self.V.grad(u_lat)-self.W.grad(u_lat,f_lat))
        return u_lat

    def adam_lat(self,u_lat_0,f_lat,lr=0.01,nmax=100):
        u_lat=nn.Parameter(u_lat_0)
        optimizer=torch.optim.Adam([u_lat],lr=lr)
        
        for i in range(nmax):
            optimizer.zero_grad()
            ener=torch.sum(self.E(u_lat,f_lat))
            ener.backward()
            optimizer.step()

        return u_lat.detach()
    
    def backward(self,f,u0=None):
        if f.dim()==2:
            batch_size=None
        else:
            batch_size=f.size(0)
        if u0 is None:
            u_lat_0=self.u_lat_0(batch_size)
        else:
            u_lat_0=self.enc_u(u0)
        f_lat=self.enc_f(f)
        u=self.dec_u(self.newton_lat(u_lat_0,f_lat))
        return u
    def force_minimum(self):
        self.V.force_minimum(self.u_lat_0(1))
        return
    



    
    
enc_f=EncoderF()
dec_f=DecoderF()