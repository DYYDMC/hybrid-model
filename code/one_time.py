def init_weights_Kaiming(m):
    if type(m) == nn.Linear or type(m) == nn.Conv3d or type(m) == nn.ConvTranspose3d\
    or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        #torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.kaiming_uniform_(m.weight)
def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)
def max_activation(model,device,xxshape,mean=0,std=1,lr=0.001,num_steps=30,test_samples=100): 
    
    # num_steps=5:optimize RF for num_steps times
    model=model.to(device)
    
    # default valus: mean=0,std=1, if use train_mean,the result is worse
    for param in model.parameters():
        param.requires_grad=False
    model=model.eval()
    if '3d' in model.__class__.__name__:
        (tempB,tempC,tempD,tempH,tempW)=xxshape#tempB should be equal to 1
        #xx=torch.randn((tempB,tempC,tempH,tempW),requires_grad=True)
        #xx=torch.zeros((tempB,tempC,tempH,tempW),requires_grad=True)
        xx=torch.zeros((tempB,tempC,tempD,tempH,tempW)).to(device)
        #xx.requires_grad=True
        ##optimizer = torch.optim.Adam([xx], lr=lr, weight_decay=0.0)
        if 'Variational' in model.__class__.__name__:
            pass
        else: # vanilla model
            out=model(xx)
            outlen=out.shape[1]
            yy=torch.zeros(outlen,tempC,tempD,tempH,tempW)
            for ii in range(outlen):
                xx=torch.zeros((tempB,tempC,tempD,tempH,tempW)).to(device)
                xx=(xx-mean)/std
                xx.requires_grad=True
                ##optimizer = torch.optim.Adam([xx], lr=lr, weight_decay=1e-6) # somehow adam does not work here
                optimizer = torch.optim.SGD([xx], lr=lr)
                for n in range(num_steps): 
                    optimizer.zero_grad()
                    out=model(xx)
                    loss = -1*out[0,ii]
                    loss.backward()
                    optimizer.step()
                    #print("epoch is {}, loss is {:.4f}".format(n, loss))
                yy[ii]=xx*std+mean
                aa=torch.linalg.norm(yy[ii])
                yy[ii]=torch.div(yy[ii],aa)                      
               
    else:    
        (tempB,tempC,tempH,tempW)=xxshape # tempB should be equal to 1
        # xx=torch.randn((tempB,tempC,tempH,tempW),requires_grad=True)
        # xx=torch.zeros((tempB,tempC,tempH,tempW),requires_grad=True)
        xx=torch.zeros((tempB,tempC,tempH,tempW)).to(device)
        # xx.requires_grad=True
        
        ## optimizer = torch.optim.Adam([xx], lr=lr, weight_decay=0.0)
        if 'Variational' in model.__class__.__name__:
            pass
        else: # vanilla model
            out=model(xx)
            outlen=out.shape[1]
            yy=torch.zeros(outlen,tempC,tempH,tempW)
            for ii in range(outlen):
                xx=torch.zeros((tempB,tempC,tempH,tempW)).to(device)
                xx=(xx-mean)/std
                xx.requires_grad=True
                ##optimizer = torch.optim.Adam([xx], lr=lr, weight_decay=1e-6) # somehow adam does not work here
                optimizer = torch.optim.SGD([xx], lr=lr)
                for n in range(num_steps): 
                    optimizer.zero_grad()
                    out=model(xx)
                    loss = -1*out[0,ii]
                    loss.backward()
                    optimizer.step()
                    #print("epoch is {}, loss is {:.4f}".format(n, loss))
                yy[ii]=xx*std+mean
                aa=torch.linalg.norm(yy[ii])
                yy[ii]=torch.div(yy[ii],aa)
    return yy