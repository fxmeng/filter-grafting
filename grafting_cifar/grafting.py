import torch
import os
import numpy as np
import collections
import time
def noise(args,net,epoch):
    model=collections.OrderedDict()
    for i,(key,u) in enumerate(net.state_dict().items()):
        if 'conv' in key:
            v=torch.norm(u,p=1,dim=(1,2,3))<args.threshold
            for j in range(len(v)):
                if v[j]:
                    u[j]+=torch.randn_like(u[j]).to(args.device)*0.975**epoch
        model[key]=u
    net.load_state_dict(model)
def inside(args,net,epoch):
    model=collections.OrderedDict()
    for i,(key,u) in enumerate(net.state_dict().items()):
        if 'conv' in key:
            v=torch.argsort(torch.norm(u,p=1,dim=(1,2,3)))
            for j in range(len(v)):
                if torch.norm(u[v[j]],p=1)<args.threshold:
                    u[v[j]]+=u[v[len(v)-j-1]]
                else:
                    break
        model[key]=u
    net.load_state_dict(model)
def entropy(x,n=10):
    x=x.reshape(-1)
    scale=(x.max()-x.min())/n
    entropy=0
    for i in range(n):
        p=torch.sum((x>=x.min()+i*scale)*(x<x.min()+(i+1)*scale),dtype=torch.float)/len(x)
        if p!=0:
            entropy-=p*torch.log(p)
    return float(entropy.cpu())
def entropy_filter(x,n=10):
    entropy=0
    for filter in x.reshape(x.shape[0],-1):
        scale=(filter.max()-filter.min())/n
        for i in range(n):
            p=torch.sum((filter>=filter.min()+i*scale)*(filter<filter.min()+(i+1)*scale),dtype=torch.float)/len(filter)
            if p!=0:
                entropy-=p*torch.log(p)
        return float(entropy.cpu())
def outsideFC(args,net,epoch):
    for i in range(args.i+1,args.i+args.num):
        while not os.access(path='./checkpoint/%s/ckpt%d_%d.t7'%(args.s,i%args.num,epoch), mode=os.R_OK):
            time.sleep(10)
    checkpoint=[]
    for i in range(args.i+1,args.i+args.num):
        try:
            checkpoint.append(torch.load('./checkpoint/%s/ckpt%d_%d.t7'%(args.s,i%args.num,epoch))['net']) 
        except:
            print('try again')
            time.sleep(10)
            checkpoint.append(torch.load('./checkpoint/%s/ckpt%d_%d.t7'%(args.s,i%args.num,epoch))['net']) 
    model=collections.OrderedDict()
    for key,u in net.state_dict().items():
        model[key]=0.25*(u+checkpoint[0][key]+checkpoint[1][key]+checkpoint[2][key])
    net.load_state_dict(model)
def outside(args,net,epoch):
    while not os.access(path='./checkpoint/%s/ckpt%d_%d.t7'%(args.s,args.i-1,epoch), mode=os.R_OK):
        time.sleep(10)
    try:
        checkpoint = torch.load('./checkpoint/%s/ckpt%d_%d.t7'%(args.s,args.i-1,epoch))['net']
    except:
        print('try again')
        time.sleep(10)
        checkpoint = torch.load('./checkpoint/%s/ckpt%d_%d.t7'%(args.s,args.i-1,epoch))['net']
    model=collections.OrderedDict()
    for i,(key,u) in enumerate(net.state_dict().items()):
        if 'conv' in key:
            if args.sh=='entropy':
                w=round(0.4/np.pi*np.arctan(args.w*(entropy(u)-entropy(checkpoint[key])))+0.5,2)
            elif args.sh=='norm':
                w=round(0.4/np.pi*np.arctan(args.w*(float(torch.norm(u,p=1).cpu())-float(torch.norm(checkpoint[key],p=1).cpu())))+0.5,2)
            print(w,end=',')
        model[key]=u*w+checkpoint[key]*(1-w)
    net.load_state_dict(model)
def grafting_filter(args,net,epoch):
    checkpoint=[]
    for i in range(args.i+1,args.i+args.num):
        while not os.access(path='./checkpoint/%s/ckpt%d_%d.t7'%(args.s,i%args.num,epoch), mode=os.R_OK):
            print('wait ckpt%d_%d.t7'%(i%args.num,epoch),end=',')
            time.sleep(10)
    for i in range(args.i+1,args.i+args.num):
        try:
            checkpoint.append(torch.load('./checkpoint/%s/ckpt%d_%d.t7'%(args.s,i%args.num,epoch))['net'])
        except:
            time.sleep(10)
            checkpoint.append(torch.load('./checkpoint/%s/ckpt%d_%d.t7'%(args.s,i%args.num,epoch))['net'])
    model=collections.OrderedDict()
    for i,(key,u) in enumerate(net.state_dict().items()):
        if 'conv' in key:
            for j,v in enumerate(u):
                if entropy(v)<args.threshold:
                    branches_index=np.argmax([entropy(checkpoint[k][key][j]) for k in range(len(checkpoint))])
                    u[j]+=checkpoint[branches_index][key][j]
        model[key]=u
    net.load_state_dict(model)