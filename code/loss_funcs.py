def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def loss_vae(recon_x, x, mu, logvar):
    tempN=x.shape[0]
    
    # elementwise mean-square-error
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    
    # Kullback–Leibler divergence, how distribution A is different from distribution B
    # A:
    # B: 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # return BCE + KLD
    
    return (MSE + KLD)/(tempN*10*2*56*56)# batch size tempN, to be comparable with other ae variants loss

# loss with L2 and L1 regularizer
# something like loss = mseloss+ alpha*alpha_x(L2)+beta*beta_y(L1), here alpha_x is a list of model layers
def loss_L2L1(recon_x, x, alpha, beta, alpha_x, beta_y):
    tempN = x.shape[0]
    MSE = F.mse_loss(recon_x, x,reduction='sum')
    l2temp=0.0
    for temp in alpha_x:
        l2temp = l2temp+ temp.weight.norm(2)
    L2loss=alpha*l2temp
    L1loss=beta*F.l1_loss(beta_y,torch.zeros_like(beta_y),reduction='sum')
    return (MSE+L2loss+L1loss)/(tempN*10*2*56*56)#batch size tempN, to be comparable with other ae variants loss

# loss with L2 and L1 regularizer, version2
# something like loss = mseloss+ alpha*alpha_x(L2)+beta*beta_y(L1), here alpha_x is a list of model layers
def loss_L2L1v2(recon_x, x, alpha, beta, alpha_x, beta_y):
    tempB, tempC, tempD, tempH, tempW =x.size()
    MSE = F.mse_loss(recon_x, x,reduction='sum')
    l2temp=0.0
    for temp in alpha_x:
        l2temp = l2temp+ temp.weight.norm(2)
    L2loss=alpha*l2temp
    B, C, D, H, W = beta_y.size() # Batch*channel*depth*height*width
    temp1=beta_y.view(B,C,-1)
    temp2=torch.norm(temp1,p=2,dim=2)
    temp3=torch.sum(torch.abs(temp2))
    L1loss=beta*temp3
    #return (MSE+L2loss+L1loss)/(tempB* 2*10*12*12)#batch size tempN, to be comparable with other ae variants loss
    return (MSE+L2loss+L1loss)/(tempB* tempC* tempD* tempH* tempW)#to be comparable with other ae variants loss

#loss with L2 and L1 regularizer, for supervised encoded
#something like loss = mseloss+ alpha*alpha_x(L2)+beta*beta_y(L1), here alpha_x is a list of model layers
def loss_L2L1_SE(recon_x, x, alpha, beta, alpha_x, beta_y):
    tempB, tempN =x.size()
    MSE = F.mse_loss(recon_x, x,reduction='sum')
    l2temp=0.0
    for temp in alpha_x:
        l2temp = l2temp+ temp.weight.norm(2)
    L2loss=alpha*l2temp
    l1temp=0.0
    for temp in beta_y:
        l1temp = l1temp+ temp.weight.norm(1)
    L1loss=beta*l1temp
    return (MSE+L2loss+L1loss)/(tempB* tempN)
#loss with L2 and L1 regularizer, for supervised encoded, Poisson loss
#something like loss = Poissonloss+ alpha*alpha_x(L2)+beta*beta_y(L1), here alpha_x is a list of model layers
def Ploss_L2L1_SE(recon_x, x, alpha, beta, alpha_x, beta_y):
    tempB, tempN =x.size()
    Ploss = F.poisson_nll_loss(recon_x, x,log_input=False, reduction='sum')
    l2temp=0.0
    for temp in alpha_x:
        l2temp = l2temp+ temp.weight.norm(2)
    L2loss=alpha*l2temp
    l1temp=0.0
    for temp in beta_y:
        l1temp = l1temp+ temp.weight.norm(1)
    L1loss=beta*l1temp
    return (Ploss+L2loss+L1loss)/(tempB* tempN)
def Ploss_L2L1_SE2(recon_x, x, alpha, alpha2, beta, alpha_x, alpha_x2, beta_y): #different convkernels with different L2 penalty
    tempB, tempN =x.size()
    Ploss = F.poisson_nll_loss(recon_x, x,log_input=False, reduction='sum')
    #
    l2temp=0.0
    for temp in alpha_x:
        l2temp = l2temp+ temp.weight.norm(2)
    L2loss=alpha*l2temp
    #
    l2temp2=0.0
    for temp in alpha_x2:
        l2temp2 = l2temp2 + temp.weight.norm(2)
    L2loss2=alpha2*l2temp2
    #
    l1temp=0.0
    for temp in beta_y:
        l1temp = l1temp+ temp.weight.norm(1)
    L1loss=beta*l1temp
    return (Ploss+L2loss+L2loss2+L1loss)/(tempB* tempN)
def Ploss_L2L1_SE_ST(recon_x, x, alpha1, alpha2, beta, alpha_x1, alpha_x2, beta_y): 
    # for spatial and temporal separable model
    tempB, tempN = x.size()
    Ploss = F.poisson_nll_loss(recon_x, x,log_input=False, reduction='sum')
    l2temp = 0.0
    for temp in alpha_x1:
        l2temp = l2temp+ temp.norm(2)
    l2temp2 = 0.0
    for temp in alpha_x2:
        l2temp2 = l2temp2+ temp.norm(2)
    L2loss = alpha1*l2temp+alpha2*l2temp2
    #
    l1temp = 0.0
    for temp in beta_y:
        l1temp = l1temp+ temp.weight.norm(1)
    L1loss = beta*l1temp
    #return (Ploss+L2loss+L1loss)/(tempB* tempN)
    return Ploss+L2loss+L1loss
def loss_L1L1_SE(recon_x, x, alpha, beta, alpha_x, beta_y):
    tempB, tempN =x.size()
    MSE = F.mse_loss(recon_x, x,reduction='sum')
    l2temp=0.0
    for temp in alpha_x:
        l2temp = l2temp+ temp.weight.norm(1)
    L2loss=alpha*l2temp
    l1temp=0.0
    for temp in beta_y:
        l1temp = l1temp+ temp.weight.norm(1)
    L1loss=beta*l1temp
    return (MSE+L2loss+L1loss)/(tempB* tempN)

#loss with L2 and L1 regularizer, for supervised encoded, L2 for conv kernel smoothness
#something like loss = mseloss+ alpha*alpha_x(L2)+beta*beta_y(L1), here alpha_x is a list of model layers
def loss_L2lapL1_SE(recon_x, x, alpha, beta, alpha_x, beta_y):
    tempB, tempN =x.size()
    MSE = F.mse_loss(recon_x, x,reduction='sum')
    l2temp=0.0
    laplacian=torch.tensor([[0.5,1.0,0.5],[1.0,-6.0,1.0],[0.5,1.0,0.5]], requires_grad=False)#laplacian kernel
    for temp in alpha_x:
        #l2temp = l2temp+ temp.weight.norm(2)
        NN,CC=temp.weight.shape[0],temp.weight.shape[1]
        laplacians=laplacian.repeat(CC, CC, 1, 1).requires_grad_(False).to(device)
        temp2=F.conv2d(temp.weight,laplacians)
        l2temp = l2temp+ temp2.norm(2)
    L2loss=alpha*l2temp
    l1temp=0.0
    for temp in beta_y:
        l1temp = l1temp+ temp.weight.norm(1)
    L1loss=beta*l1temp
    return (MSE+L2loss+L1loss)/(tempB* tempN)