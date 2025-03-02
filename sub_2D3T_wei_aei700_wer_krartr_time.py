
import torch
from torch.autograd import Variable as V
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from matplotlib.ticker import MaxNLocator
import argparse
import os
import pdb
# torch.set_default_tensor_type(torch.DoubleTensor)

parser = argparse.ArgumentParser('xPINN_Poissons_2D')
parser.add_argument('--Number', type=int, default=1)
parser.add_argument('--BATCH_SIZE', type=int, default=pow(2, 5))

parser.add_argument('--n_hyper', type=float, default=100)  # the slope used in adaptive activation function
parser.add_argument('--lr', type=float, default=0.001)    # learning rate
parser.add_argument('--lr_decay', type=float, default=0.999)    # the coefficient of learning rate decay
#parser.add_argument('--lr_decay', type=float, default=1)    # the coefficient of learning rate decay
parser.add_argument('--dt', type=float, default=0.2)     # the step length in skip connection
parser.add_argument('--which_act', type=int, default=2)    # activation: 1-ReLU, 2-Tanh, 3-Sigmoid, 4-LeakyReLU, 5-GELU
parser.add_argument('--which_loss', type=int, default=1)   # loss: 1-MSELoss, 2-L1Loss, 3-SmoothL1Loss
parser.add_argument('--which_decay', type=int, default=1)  # lr_decay:1-ExponentialLR, 2-StepLR, 3-MultiStepLR
parser.add_argument('--weight_decay', type=float, default=0)    # weight decay in optimizer

parser.add_argument('--GPU', type=int, default=0)
# parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()      # parsing the parameters
argsDict = args.__dict__


# ===========================================================
# ****  这里加了一个控制设备和精度的，拿GPU跑了一个100000 iteration的
# ===========================================================
# DeviceDtype = {'device':'cpu',
#                 'dtype':torch.float64}
DeviceDtype = {'device':'cuda',
                'dtype':torch.float64}
# FolderName = 'Train_10000_20000_04131400/'

class Block(nn.Module):
    def __init__(self, n_hidden, coe):
        super(Block, self).__init__()
        self.block = self.build_block(n_hidden)
        #slope是斜率，也是要学习的参数
        self.slope = torch.nn.Parameter(torch.Tensor([coe]))

    def build_block(self, n_hidden, which_act=args.which_act):
        block = []
        block += [nn.Linear(n_hidden, n_hidden)]

        if which_act == 1:
            block += [nn.ReLU()]
        elif which_act == 2:
            block += [nn.Tanh()]
        elif which_act == 3:
            block += [nn.Sigmoid()]
        elif which_act == 4:
            block += [nn.LeakyReLU(0.1)]
        elif which_act == 5:
            block += [nn.GELU()]
            # print("gelu")
        return nn.Sequential(*block)

    def forward(self, x):
        #Resnet也加上
        # return (x + args.dt * self.block(x))*self.slope*args.n_hyper
        return (self.block(x))*self.slope*args.n_hyper
        # return (self.block(x))


class PhysicsInformedNN:
    # Initialize the class
 
    # def __init__(self, Xlbtrain, Xubtrain, Ylbtrain, Yubtrain, T0train, u0, X_f_train_total, NN_layers_total,\
    #              rou, ce, ci, cr, beta, Ae, Ai, Ar, Aei, Aer):
 
    def __init__(self, Xlbtrain, Xubtrain, Ylbtrain, Yubtrain, T0train, X_f_train_total, NN_layers_total,\
                  rou, ce, ci, cr, beta, Ae, Ai, Ar, Aei, Aer, lrin=None):

        self.xlbtrain = Xlbtrain
        self.xubtrain = Xubtrain
        self.ylbtrain = Ylbtrain
        self.yubtrain = Yubtrain
        self.t0train = T0train
        # self.ut0 = u0
        self.x_f_total = X_f_train_total
        self.layers = NN_layers_total
        self.rou = rou
        self.ce = ce
        self.ci = ci
        self.cr = cr
        self.beta = beta
        self.Ae = Ae
        self.Ai = Ai
        self.Ar = Ar
        self.Aei = Aei
        self.Aer = Aer
        if lrin:
            args.lr=lrin


        self.x_xlb = torch.from_numpy(Xlbtrain[:, 0:1]).to(**DeviceDtype)
        self.y_xlb = torch.from_numpy(Xlbtrain[:, 1:2]).to(**DeviceDtype)
        self.t_xlb = torch.from_numpy(Xlbtrain[:, 2:3]).to(**DeviceDtype)
        self.x_xub = torch.from_numpy(Xubtrain[:, 0:1]).to(**DeviceDtype)
        self.y_xub = torch.from_numpy(Xubtrain[:, 1:2]).to(**DeviceDtype)
        self.t_xub = torch.from_numpy(Xubtrain[:, 2:3]).to(**DeviceDtype)
        self.x_ylb = torch.from_numpy(Ylbtrain[:, 0:1]).to(**DeviceDtype)
        self.y_ylb = torch.from_numpy(Ylbtrain[:, 1:2]).to(**DeviceDtype)
        self.t_ylb = torch.from_numpy(Ylbtrain[:, 2:3]).to(**DeviceDtype)
        self.x_yub = torch.from_numpy(Yubtrain[:, 0:1]).to(**DeviceDtype)
        self.y_yub = torch.from_numpy(Yubtrain[:, 1:2]).to(**DeviceDtype)
        self.t_yub = torch.from_numpy(Yubtrain[:, 2:3]).to(**DeviceDtype)
        
        self.x_t0 = torch.from_numpy(T0train[:, 0:1]).to(**DeviceDtype)
        self.y_t0 = torch.from_numpy(T0train[:, 1:2]).to(**DeviceDtype)
        self.t_t0 = torch.from_numpy(T0train[:, 2:3]).to(**DeviceDtype)
        # self.Te_t0 = torch.from_numpy(u0[:, 0:1]).to(**DeviceDtype)
        # self.Ti_t0 = torch.from_numpy(u0[:, 1:2]).to(**DeviceDtype)
        # self.Tr_t0 = torch.from_numpy(u0[:, 2:3]).to(**DeviceDtype)

        
        self.x_f = torch.from_numpy(X_f_train_total[:, 0:1]).to(**DeviceDtype)
        self.y_f = torch.from_numpy(X_f_train_total[:, 1:2]).to(**DeviceDtype)
        self.t_f = torch.from_numpy(X_f_train_total[:, 2:3]).to(**DeviceDtype)
        
        lb = np.array([0, 0, 0])
        ub = np.array([1, 1, 1.1])
        self.lb = torch.from_numpy(lb).to(**DeviceDtype)
        self.ub = torch.from_numpy(ub).to(**DeviceDtype)
        #self.mu_ = torch.tensor([400.0], requires_grad=True).to(**DeviceDtype)
        #self.mu_ = torch.nn.Parameter(self.mu_)

        size = 1000
        #net_u是网络，每个进程网络结构不同
        layers = [nn.Linear(in_features=self.layers[0][0]+2*size, out_features=self.layers[0][1]), nn.Tanh()]
        #layers = [nn.Linear(in_features=self.layers[0][0], out_features=self.layers[0][1]), nn.Tanh()]
        self.slope = torch.nn.Parameter(torch.ones(len(self.layers[0])-2)*1/args.n_hyper)
        for i in range(len(self.layers[0])-2):
            layers += [Block(self.layers[0][i+1], self.slope[i])]
        layers += [nn.Linear(self.layers[0][-2], self.layers[0][-1])]

        self.net_u = nn.Sequential(*layers).to(**DeviceDtype)
        #self.net_u.register_parameter('mu_', self.mu_)
        self.FF_weight = torch.randn((self.layers[0][0],size)).to(**DeviceDtype)


    #改一下非负激活函数
    #net只是对输入数据先映射到[-1,1]，再进入net_u
    def net(self, x):
        x = 2.0*(x - self.lb)/(self.ub - self.lb)-1.0
        y = x@self.FF_weight  # N x size
        x = torch.cat((x,torch.cos(y),torch.sin(y)),1)
        # return self.net_u(x)
        return torch.log(1+torch.exp(self.net_u(x)))
        # return 1/2*torch.log(1+torch.exp(1/2*self.net_u(x)))+3e-4
        #1/2ln(1+e(1/2*x))
        #gelu
        #其他边界条件都不加
        #找其他合适的参数,离子温度要先上去才行，才能解现象
        #从时间往后推，看能不能解,看哪个时刻能解好
    
    def _grad_(self, y, x):
        return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True, retain_graph=True)
    
    def net_dtdnei(self, x_xlb,y_xlb,t_xlb,x_xub,y_xub,t_xub,\
                        x_ylb,y_ylb,t_ylb,x_yub,y_yub,t_yub):

               
        x_xlb = V(x_xlb, requires_grad=True)
        y_xlb = V(y_xlb, requires_grad=True)
        t_xlb = V(t_xlb, requires_grad=True)
        x_xub = V(x_xub, requires_grad=True)
        y_xub = V(y_xub, requires_grad=True)
        t_xub = V(t_xub, requires_grad=True)
        x_ylb = V(x_ylb, requires_grad=True)
        y_ylb = V(y_ylb, requires_grad=True)
        t_ylb = V(t_ylb, requires_grad=True)
        x_yub = V(x_yub, requires_grad=True)
        y_yub = V(y_yub, requires_grad=True)
        t_yub = V(t_yub, requires_grad=True)
        
        Teg1, Tig1, Trg1 = torch.split(self.net(torch.cat([x_ylb,y_ylb,t_ylb], 1)),1,1)
        Teg1_x, Teg1_y, Teg1_t = self._grad_(Teg1, [x_ylb,y_ylb,t_ylb])
        Tig1_x, Tig1_y, Tig1_t = self._grad_(Tig1, [x_ylb,y_ylb,t_ylb])
        
        Teg2, Tig2, Trg2 = torch.split(self.net(torch.cat([x_xub,y_xub,t_xub], 1)),1,1)
        Teg2_x, Teg2_y, Teg2_t = self._grad_(Teg2, [x_xub,y_xub,t_xub])
        Tig2_x, Tig2_y, Tig2_t = self._grad_(Tig2, [x_xub,y_xub,t_xub])
        
        Teg3, Tig3, Trg3 = torch.split(self.net(torch.cat([x_yub,y_yub,t_yub], 1)),1,1)
        Teg3_x, Teg3_y, Teg3_t = self._grad_(Teg3, [x_yub,y_yub,t_yub])
        Tig3_x, Tig3_y, Tig3_t = self._grad_(Tig3, [x_yub,y_yub,t_yub])
        
        Teg4, Tig4, Trg4 = torch.split(self.net(torch.cat([x_xlb,y_xlb,t_xlb], 1)),1,1)
        Teg4_x, Teg4_y, Teg4_t = self._grad_(Teg4, [x_xlb,y_xlb,t_xlb])
        Tig4_x, Tig4_y, Tig4_t = self._grad_(Tig4, [x_xlb,y_xlb,t_xlb])
        
    
        return Teg1_y, Teg2_x, Teg3_y, Teg4_x, Tig1_y, Tig2_x, Tig3_y, Tig4_x
    def net_dtdnr(self,x_xlb,y_xlb,t_xlb,x_xub,y_xub,t_xub,\
                        x_ylb,y_ylb,t_ylb):
        
        x_xlb = V(x_xlb, requires_grad=True)
        y_xlb = V(y_xlb, requires_grad=True)
        t_xlb = V(t_xlb, requires_grad=True)
        x_xub = V(x_xub, requires_grad=True)
        y_xub = V(y_xub, requires_grad=True)
        t_xub = V(t_xub, requires_grad=True)
        x_ylb = V(x_ylb, requires_grad=True)
        y_ylb = V(y_ylb, requires_grad=True)
        t_ylb = V(t_ylb, requires_grad=True)
        
        Teg1, Tig1, Trg1 = torch.split(self.net(torch.cat([x_ylb,y_ylb,t_ylb], 1)),1,1)
        Trg1_x, Trg1_y, Trg1_t = self._grad_(Trg1, [x_ylb,y_ylb,t_ylb])
        
        Teg2, Tig2, Trg2 = torch.split(self.net(torch.cat([x_xub,y_xub,t_xub], 1)),1,1)
        Trg2_x, Trg2_y, Trg2_t = self._grad_(Trg2, [x_xub,y_xub,t_xub])
        
        Teg4, Tig4, Trg4 = torch.split(self.net(torch.cat([x_xlb,y_xlb,t_xlb], 1)),1,1)
        Trg4_x, Trg4_y, Trg4_t = self._grad_(Trg4, [x_xlb,y_xlb,t_xlb])
        
        
        return Trg1_y, Trg2_x, Trg4_x
            
    
    def net_f(self, x, y, t, rou, ce, ci, cr, beta, Ae, Ai, Ar, Aei, Aer):
        x = V(x, requires_grad=True)
        y = V(y, requires_grad=True)
        t = V(t, requires_grad=True)
        
        Te, Ti, Tr = torch.split(self.net(torch.cat([x, y, t], 1)),1,1)

        Te_x, Te_y, Te_t = self._grad_(Te, [x,y,t])
        Ti_x, Ti_y, Ti_t = self._grad_(Ti, [x,y,t])
        Tr_x, Tr_y, Tr_t = self._grad_(Tr, [x,y,t])
        
        # Ke, Ki, Kr = Ae*Te.pow(5/2),Ai*Ti.pow(5/2),Ar*Tr.pow(3+beta)
        # Ke, Ki, Kr = Ae*Te.pow(5/2),Ai*Ti.pow(5/2),10
        #Ke, Ki, Kr = Ae*Te.pow(5/2),Ai*Ti.pow(5/2),Ar
        Ke, Ki, Kr = Ae*Te.pow(5/2),Ai*Ti.pow(5/2),Ar*Tr
        # Ke, Ki, Kr = 10, 10, 10
        # Ke, Ki, Kr = 1, 1, 1
        wei, wer = rou*Aei*Te.pow(-2/3), rou*Aer*Te.pow(-1/2)
        # wei, wer = 1, 1
        # wei, wer = 10, 10
        # wei, wer = 10, rou*Aer*Te.pow(-1/2)
        # wei, wer = rou*Aei*Te.pow(-2/3), 10
        # wei, wer = 0, 0
        Nb_e = self._grad_(Ke*Te_x,x)[0]+self._grad_(Ke*Te_y,y)[0]
        Nb_i = self._grad_(Ki*Ti_x,x)[0]+self._grad_(Ki*Ti_y,y)[0]
        Nb_r = self._grad_(Kr*Tr_x,x)[0]+self._grad_(Kr*Tr_y,y)[0]
        
        
        feleft = ce*Te_t-1/rou*Nb_e
        feright = wei*(Ti-Te) + wer*(Tr-Te)
        fe = feleft - feright
        
        fileft = ci*Ti_t-1/rou*Nb_i
        firight = wei*(Te-Ti)
        fi = fileft - firight
        
        frleft = cr*Tr.pow(3)*Tr_t-1/rou*Nb_r
        frright = wer*(Te-Tr)
        fr = frleft - frright      

        return fe,fi,fr



    # def train(self, nIter, X_star, u_star):
    def train(self, nIter, X_star):

        opt = torch.optim.Adam(self.net_u.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        opt_lbfgs = torch.optim.LBFGS(self.net_u.parameters(), lr=1.2, tolerance_change=1e-302, tolerance_grad=1e-302,
                                      history_size=120, max_iter=30000, line_search_fn='strong_wolfe')
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_decay)

        if args.which_loss == 1:
            loss_func = torch.nn.MSELoss()
        elif args.which_loss == 2:
            loss_func = torch.nn.L1Loss()
        else:
            loss_func = torch.nn.SmoothL1Loss()

        Temseloss = []
        Timseloss = []
        Trmseloss = []
        MSE_history = []
        a_history = []
        L2error_u = []
        TeL2 = []
        TiL2 = []
        TrL2 = []
        L1error_u = []
        Linferror_u = []
        Teiniloss = []
        Tiiniloss = []
        Triniloss = []
        Trboundloss = []
        lr = []
        
        
        def Tr0(y):
            u = (10.0 - 2.0) * np.exp(10.0 * (y - 1)) + 2.0
            return u
        
        def Trfree_boundary(x,t):
            # u = 10.0*np.ones_like(t)
            # u = 3e-4 + 0.2*t
            u = 3e-4 + 2*t
            return u
        
        # def format_fn(value, tick_number): 
        #     return f"{value:.4f}"
        

        # ============================================================
        # ****  Define Loss and Logging Information  
        # ============================================================ 

        def complus_loss(it = None):
            
            #把训练点代入网络得到预测值        
            u0_pred = self.net(torch.cat((self.x_t0, self.y_t0, self.t_t0), 1))
            
            #上边界是自由面边界
            urg3_pred = self.net(torch.cat((self.x_yub, self.y_yub, self.t_yub), 1))
            
            #因为都是趋于0，所以正负号不重要，但是法向应该有正负号吧！

            # 这里用3和8结尾 标记了一下长度
            ret_Tr3 = self.net_dtdnr(self.x_xlb,self.y_xlb,self.t_xlb,self.x_xub,self.y_xub,self.t_xub,
                                    self.x_ylb,self.y_ylb,self.t_ylb)
            
            ret_TeTi8 = self.net_dtdnei(self.x_xlb,self.y_xlb,self.t_xlb,self.x_xub,self.y_xub,self.t_xub,
                                        self.x_ylb,self.y_ylb,self.t_ylb,self.x_yub,self.y_yub,self.t_yub)
            
            
                    
            # ret_f = self.net_f(self.x_f, self.y_f, self.t_f, self.rou, self.ce, self.ci, self.cr, self.beta, self.Ae, self.Ai, self.Ar, self.Aei, self.Aer)
            
            fe,fi,fr = ret_f = self.net_f(self.x_f, self.y_f, self.t_f, self.rou, self.ce, self.ci, self.cr, self.beta, self.Ae, self.Ai, self.Ar, self.Aei, self.Aer)
            

            t0=3e-4
            # dr=10.0
            
            #把前三个改一下，读文件就行
            #mse损失 平方和取平均
            """
            losste = 100*loss_func(u0_pred[:,0:1],t0*torch.ones_like(u0_pred[:,0:1])) +\
                1*sum([loss_func(TT,torch.zeros_like(TT)) for TT in ret_TeTi8[:4]]) +\
                    1*loss_func(fe,torch.zeros_like(fe))
                    
            lossti = 100*loss_func(u0_pred[:,1:2],t0*torch.ones_like(u0_pred[:,1:2])) +\
                1*sum([loss_func(TT,torch.zeros_like(TT)) for TT in ret_TeTi8[4:]]) +\
                    1*loss_func(fi,torch.zeros_like(fi))
                    
            losstr = 100*loss_func(u0_pred[:,2:3],t0*torch.ones_like(u0_pred[:,2:3])) +\
                1000*loss_func(urg3_pred[:,2:3],Trfree_boundary(self.x_yub, self.t_yub))+\
                    1*sum([loss_func(TT,torch.zeros_like(TT)) for TT in ret_Tr3]) +\
                    1*loss_func(fr,torch.zeros_like(fr))
                    
            loss = 1*lossti + 1*losste + 1*losstr
            """
            # loss = 10*loss_func(u0_pred[:,0:1],t0*torch.ones_like(u0_pred[:,0:1])) +\
            #     10*loss_func(u0_pred[:,1:2],t0*torch.ones_like(u0_pred[:,1:2])) +\
            #         10*loss_func(u0_pred[:,2:3],t0*torch.ones_like(u0_pred[:,2:3])) +\
            #             1000*loss_func(urg3_pred[:,2:3],Trfree_boundary(self.x_yub, self.t_yub))+\
            #                 1*sum([loss_func(TT,torch.zeros_like(TT)) for TT in ret_Tr3+ret_TeTi8]) +\
            #                     1*sum([loss_func(FF,torch.zeros_like(FF)) for FF in ret_f])
            
            #可能aei=400 kr=ar用的就是这个？
            # loss = 10*loss_func(u0_pred[:,0:1],t0*torch.ones_like(u0_pred[:,0:1])) +\
            #     10*loss_func(u0_pred[:,1:2],t0*torch.ones_like(u0_pred[:,1:2])) +\
            #         10*loss_func(u0_pred[:,2:3],t0*torch.ones_like(u0_pred[:,2:3])) +\
            #             20*loss_func(urg3_pred[:,2:3],Trfree_boundary(self.x_yub, self.t_yub))+\
            #                 1*sum([loss_func(TT,torch.zeros_like(TT)) for TT in ret_Tr3+ret_TeTi8]) +\
            #                     1*sum([loss_func(FF,torch.zeros_like(FF)) for FF in ret_f])
            
            #对ke ki非线性可解的loss
            # loss = 1*loss_func(torch.log(u0_pred[:,0:1]),torch.log(t0*torch.ones_like(u0_pred[:,0:1]))) +\
            #     1*loss_func(torch.log(u0_pred[:,1:2]),torch.log(t0*torch.ones_like(u0_pred[:,1:2]))) +\
            #         1*loss_func(torch.log(u0_pred[:,2:3]),torch.log(t0*torch.ones_like(u0_pred[:,2:3]))) +\
            #             1000*loss_func(urg3_pred[:,2:3],Trfree_boundary(self.x_yub, self.t_yub))+\
            #                 1*sum([loss_func(TT,torch.zeros_like(TT)) for TT in ret_Tr3+ret_TeTi8]) +\
            #                     10*sum([loss_func(FF,torch.zeros_like(FF)) for FF in ret_f])
            """
            lossmu = (self.mu_ - self.Aei)**2            
            ##Aei=400,Kr=Ar用的这个权重！！！
            loss = 10*loss_func(torch.log(u0_pred[:,0:1]),torch.log(t0*torch.ones_like(u0_pred[:,0:1]))) +\
                10*loss_func(torch.log(u0_pred[:,1:2]),torch.log(t0*torch.ones_like(u0_pred[:,1:2]))) +\
                    10*loss_func(torch.log(u0_pred[:,2:3]),torch.log(t0*torch.ones_like(u0_pred[:,2:3]))) +\
                        20*loss_func(torch.log(urg3_pred[:,2:3]),torch.log(Trfree_boundary(self.x_yub, self.t_yub)))+\
                            1*sum([loss_func(TT,torch.zeros_like(TT)) for TT in ret_Tr3+ret_TeTi8]) +\
                                1*sum([loss_func(FF,torch.zeros_like(FF)) for FF in ret_f])+\
				    lossmu
	    """
            lossfe = loss_func(fe,torch.zeros_like(fe))
            lossfi = loss_func(fi,torch.zeros_like(fi))
            lossfr = loss_func(fr,torch.zeros_like(fr))
            loss = 10*loss_func(torch.log(u0_pred[:,0:1]),torch.log(t0*torch.ones_like(u0_pred[:,0:1]))) +\
                10*loss_func(torch.log(u0_pred[:,1:2]),torch.log(t0*torch.ones_like(u0_pred[:,1:2]))) +\
                    10*loss_func(torch.log(u0_pred[:,2:3]),torch.log(t0*torch.ones_like(u0_pred[:,2:3]))) +\
                        20*loss_func(torch.log(urg3_pred[:,2:3]),torch.log(Trfree_boundary(self.x_yub, self.t_yub)))+\
                            1*sum([loss_func(TT,torch.zeros_like(TT)) for TT in ret_Tr3]) +\
                                1*sum([loss_func(TT,torch.zeros_like(TT)) for TT in ret_TeTi8]) +\
                                    lossfe + lossfi + lossfr
            #aei=400 kr=ar用的就是这个
            # loss = 10*loss_func(torch.log(u0_pred[:,0:1]),torch.log(t0*torch.ones_like(u0_pred[:,0:1]))) +\
            #     10*loss_func(torch.log(u0_pred[:,1:2]),torch.log(t0*torch.ones_like(u0_pred[:,1:2]))) +\
            #         10*loss_func(torch.log(u0_pred[:,2:3]),torch.log(t0*torch.ones_like(u0_pred[:,2:3]))) +\
            #             20*loss_func(torch.log(urg3_pred[:,2:3]),torch.log(Trfree_boundary(self.x_yub, self.t_yub)))+\
            #                 1*sum([loss_func(TT,torch.zeros_like(TT)) for TT in ret_Tr3]) +\
            #                     1*sum([loss_func(TT,torch.zeros_like(TT)) for TT in ret_TeTi8]) +\
            #                         1*sum([loss_func(FF,torch.zeros_like(FF)) for FF in ret_f])
            #aei=450 kr=ar用的就是这个
            """
            loss = 10*loss_func(torch.log(u0_pred[:,0:1]),torch.log(t0*torch.ones_like(u0_pred[:,0:1]))) +\
                10*loss_func(torch.log(u0_pred[:,1:2]),torch.log(t0*torch.ones_like(u0_pred[:,1:2]))) +\
                    10*loss_func(torch.log(u0_pred[:,2:3]),torch.log(t0*torch.ones_like(u0_pred[:,2:3]))) +\
                        20*loss_func(torch.log(urg3_pred[:,2:3]),torch.log(Trfree_boundary(self.x_yub, self.t_yub)))+\
                            1*sum([loss_func(TT,torch.zeros_like(TT)) for TT in ret_Tr3]) +\
                                10*sum([loss_func(TT,torch.zeros_like(TT)) for TT in ret_TeTi8]) +\
                                    1*sum([loss_func(FF,torch.zeros_like(FF)) for FF in ret_f])
	    """
            # loss = 10*loss_func(torch.log(u0_pred[:,0:1]),torch.log(t0*torch.ones_like(u0_pred[:,0:1]))) +\
            #     10*loss_func(torch.log(u0_pred[:,1:2]),torch.log(t0*torch.ones_like(u0_pred[:,1:2]))) +\
            #         10*loss_func(torch.log(u0_pred[:,2:3]),torch.log(t0*torch.ones_like(u0_pred[:,2:3]))) +\
            #             1000*loss_func(torch.log(urg3_pred[:,2:3]),torch.log(Trfree_boundary(self.x_yub, self.t_yub)))+\
            #                 1*sum([loss_func(TT,torch.zeros_like(TT)) for TT in ret_Tr3+ret_TeTi8]) +\
            #                     1*sum([loss_func(FF,torch.zeros_like(FF)) for FF in ret_f])
                                
            # loss = 10*loss_func(torch.log(u0_pred[:,0:1]),torch.log(t0*torch.ones_like(u0_pred[:,0:1]))) +\
            #     10*loss_func(torch.log(u0_pred[:,1:2]),torch.log(t0*torch.ones_like(u0_pred[:,1:2]))) +\
            #         10*loss_func(torch.log(u0_pred[:,2:3]),torch.log(t0*torch.ones_like(u0_pred[:,2:3]))) +\
            #             1000*loss_func(torch.log(urg3_pred[:,2:3]),torch.log(Trfree_boundary(self.x_yub, self.t_yub)))+\
            #                 1*sum([torch.log(loss_func(TT,torch.zeros_like(TT))) for TT in ret_Tr3+ret_TeTi8]) +\
            #                     1*sum([torch.log(loss_func(FF,torch.zeros_like(FF))) for FF in ret_f])
                                
            # loss = 10*loss_func(torch.log(u0_pred[:,0:1]),torch.log(t0*torch.ones_like(u0_pred[:,0:1]))) +\
            #     10*loss_func(torch.log(u0_pred[:,1:2]),torch.log(t0*torch.ones_like(u0_pred[:,1:2]))) +\
            #         10*loss_func(torch.log(u0_pred[:,2:3]),torch.log(t0*torch.ones_like(u0_pred[:,2:3]))) +\
            #             1000*loss_func(torch.log(urg3_pred[:,2:3]),torch.log(Trfree_boundary(self.x_yub, self.t_yub)))+\
            #                 1*sum([loss_func(TT,torch.zeros_like(TT)) for TT in ret_Tr3+ret_TeTi8]) +\
            #                     10*sum([loss_func(FF,torch.zeros_like(FF)) for FF in ret_f])
            
            # loss = 10*loss_func(u0_pred[:,0:1],self.Te_t0) +\
            #     10*loss_func(u0_pred[:,1:2],self.Ti_t0) +\
            #         10*loss_func(u0_pred[:,2:3],self.Tr_t0) +\
            #             10*loss_func(urg3_pred[:,2:3],dr*torch.ones_like(urg3_pred[:,2:3]))+\
            #                 20*sum([loss_func(TT,torch.zeros_like(TT)) for TT in ret_Tr3+ret_TeTi8]) +\
            #                     20*sum([loss_func(FF,torch.zeros_like(FF)) for FF in ret_f])
                  
            # loss = 10*loss_func(u0_pred[:,0:1],t0*torch.ones_like(u0_pred[:,0:1])) +\
            #     10*loss_func(u0_pred[:,1:2],t0*torch.ones_like(u0_pred[:,1:2])) +\
            #         10*loss_func(u0_pred[:,2:3],t0*torch.ones_like(u0_pred[:,2:3])) +\
            #             10*loss_func(urg3_pred[:,2:3],dr*torch.ones_like(urg3_pred[:,2:3]))+\
            #                 20*sum([loss_func(TT,torch.zeros_like(TT)) for TT in ret_Tr3+ret_TeTi8]) +\
            #                     20*sum([loss_func(FF,torch.zeros_like(FF)) for FF in ret_f])
                                
            # print(Trfree_boundary(self.x_yub, self.t_yub))
            

            if it == None:
                # LBFGS 有两种 “迭代次数”，一个是权重更新次数，一个是Loss函数的预测次数。
                # 但是第一种（权重更新次数）可能会导致重复记录，因为多次一次更新可能多次调用Loss，导致重复被记录
                # it = opt_lbfgs.state[opt.param_groups[0]['params'][0]]['n_iter']
                it = opt_lbfgs.state[opt.param_groups[0]['params'][0]]['func_evals']
            if lossfe < 1e-3 and lossfi < 1e-3 and lossfr < 1e-3 == 0:
                return lossfe, lossfi, lossfr
            """
            if it % 200 == 0:
                Tep_pred,Tip_pred,Trp_pred = torch.split(self.predict(X_star),1,1)
                Tep_pred = Tep_pred.data.cpu().numpy().reshape(-1,1)
                Tip_pred = Tip_pred.data.cpu().numpy().reshape(-1,1)
                Trp_pred = Trp_pred.data.cpu().numpy().reshape(-1,1)
                
                # Tel2_err = np.linalg.norm(u_star[:,0:1]-Tep_pred,2)/np.linalg.norm(u_star[:,0:1],2) 
                # Til2_err = np.linalg.norm(u_star[:,1:2]-Tip_pred,2)/np.linalg.norm(u_star[:,1:2],2)
                # Trl2_err = np.linalg.norm(u_star[:,2:3]-Trp_pred,2)/np.linalg.norm(u_star[:,2:3],2)
                

                # Tel1_err = np.linalg.norm(u_star[:,0:1]-Tep_pred,1)/625
                # Til1_err = np.linalg.norm(u_star[:,1:2]-Tip_pred,1)/625
                # Trl1_err = np.linalg.norm(u_star[:,2:3]-Trp_pred,1)/625
                
                # Telinf_err = np.linalg.norm(u_star[:,0:1]-Tep_pred,np.inf)
                # Tilinf_err = np.linalg.norm(u_star[:,1:2]-Tip_pred,np.inf)
                # Trlinf_err = np.linalg.norm(u_star[:,2:3]-Trp_pred,np.inf)
                
                fe_pred,fi_pred,fr_pred = ret_f
                brg1_pred,brg2_pred,brg4_pred = ret_Tr3
                bei8 = sum([loss_func(TT,torch.zeros_like(TT)) for TT in ret_TeTi8])
                be4 = sum([loss_func(TT,torch.zeros_like(TT)) for TT in ret_TeTi8[:4]])
                bi4 = sum([loss_func(TT,torch.zeros_like(TT)) for TT in ret_TeTi8[4:]])
                # print('It: %d, Loss:%.3e' %
                #       (it, Trfree_boundary(self.x_yub, self.t_yub)))
                print("It: %d, after scheduler step ,lr is: %.6e" % (it, scheduler.get_last_lr()[0]))
                print('Aei: %d, It: %d, be4: %.3e, bi4: %.3e' %
                      (self.Aei, it, be4, bi4))
                #print('Aei: %d, It: %d, be4: %.3e, bi4: %.3e, mu_: %.3e' %
                #      (self.Aei, it, be4, bi4, self.mu_))
                # print('It: %d, Loss:%.3e' %
                #       (it, Trfree_boundary(self.x_yub, self.t_yub)))
                print('It: %d, Loss:%.3e,\nfeloss: %.3e, filoss: %.3e, frloss: %.3e, \
                      inieloss: %.3e, iniiloss: %.3e, inirloss: %.3e, \
                      borg3: %.3e, borg1: %.3e, borg2: %.3e, borg4:%.3e, \
                      bei8: %.3e' % 
                # print('It: %d, Loss:%.3e,\nfeloss: %.3e, filoss: %.3e, frloss: %.3e, \
                #       inieloss: %.3e, iniiloss: %.3e, inirloss: %.3e, \
                #       borg1: %.3e, borg2: %.3e, borg4:%.3e, \
                #       bei8: %.3e' % 
                      (it, loss, loss_func(fe_pred, torch.zeros_like(fe_pred)).cpu().data.numpy(),
                      loss_func(fi_pred, torch.zeros_like(fi_pred)).cpu().data.numpy(),
                      loss_func(fr_pred, torch.zeros_like(fr_pred)).cpu().data.numpy(),
                      loss_func(u0_pred[:,0:1],t0*torch.ones_like(u0_pred[:,0:1])).cpu().data.numpy(),
                      loss_func(u0_pred[:,1:2],t0*torch.ones_like(u0_pred[:,1:2])).cpu().data.numpy(),
                      # loss_func(u0_pred[:,2:3],Tr0(self.y_t0)).cpu().data.numpy(),
                      loss_func(u0_pred[:,2:3],t0*torch.ones_like(u0_pred[:,2:3])).cpu().data.numpy(),
                      # loss_func(u0_pred[:,0:1],self.Te_t0).cpu().data.numpy(),
                      # loss_func(u0_pred[:,1:2],self.Ti_t0).cpu().data.numpy(),
                      # loss_func(u0_pred[:,2:3],self.Tr_t0).cpu().data.numpy(),
                      # loss_func(urg3_pred[:,2:3],dr*torch.ones_like(urg3_pred[:,2:3])).cpu().data.numpy(),
                      loss_func(urg3_pred[:,2:3],Trfree_boundary(self.x_yub, self.t_yub)).cpu().data.numpy(),
                      loss_func(brg1_pred, torch.zeros_like(brg1_pred)).cpu().data.numpy(),
                      loss_func(brg2_pred, torch.zeros_like(brg2_pred)).cpu().data.numpy(),
                      loss_func(brg4_pred, torch.zeros_like(brg4_pred)).cpu().data.numpy(),
                      bei8))
                # print('It: %d,  Loss: %.3e,\n TeL2_err: %.3e, TeL1_err: %.3e, TeLinf_err: %.3e,\n TiL2_err: %.3e, TiL1_err: %.3e, TiLinf_err: %.3e,\n TrL2_err: %.3e, TrL1_err: %.3e, TrLinf_err: %.3e'\
                #       %  (it, loss, Tel2_err, Tel1_err, Telinf_err, Til2_err, Til1_err, Tilinf_err, Trl2_err, Trl1_err, Trlinf_err))
                Temseloss.append(loss_func(fe_pred, torch.zeros_like(fe_pred)).cpu().data.numpy())
                Timseloss.append(loss_func(fi_pred, torch.zeros_like(fi_pred)).cpu().data.numpy())
                Trmseloss.append(loss_func(fr_pred, torch.zeros_like(fr_pred)).cpu().data.numpy())
                
                # TeL2.append(Tel2_err)
                # TiL2.append(Til2_err)
                # TrL2.append(Trl2_err)
                
                Teiniloss.append(loss_func(u0_pred[:,0:1],t0*torch.ones_like(u0_pred[:,0:1])).cpu().data.numpy())
                Tiiniloss.append(loss_func(u0_pred[:,1:2],t0*torch.ones_like(u0_pred[:,1:2])).cpu().data.numpy())
                Triniloss.append(loss_func(u0_pred[:,2:3],t0*torch.ones_like(u0_pred[:,2:3])).cpu().data.numpy())
                Trboundloss.append(loss_func(urg3_pred[:,2:3],Trfree_boundary(self.x_yub, self.t_yub)).cpu().data.numpy())
                
                lr.append(scheduler.get_last_lr()[0])
                
                # Teiniloss.append(loss_func(u0_pred[:,0:1],self.Te_t0).cpu().data.numpy())
                # Tiiniloss.append(loss_func(u0_pred[:,1:2],self.Ti_t0).cpu().data.numpy())
                # Triniloss.append(loss_func(u0_pred[:,2:3],self.Tr_t0).cpu().data.numpy())
            """
            """
            if it % 1000 == 0:
                def format_fn(value, tick_number): 
                    return f"{value:.4f}"
                
                FolderName = 'Train_6000_30000_11281214/'
                
                with open('./sol1_wei_aei700_wer_krartr_1.txt','r') as fp:
                # with open('./sol1e-8.txt','r') as fp:
                    import numpy as np
                    from itertools import product
                    import re
                    DIM, Num_T, _ = eval(fp.readline())
                    print(DIM, Num_T, _)
                    (Imin,Jmin),(Imax,Jmax),_ = eval(fp.readline())
                    print((Imin,Jmin),(Imax,Jmax),_)
                    Data = np.zeros((Imax-Imin+1, Jmax-Jmin+1, Num_T))
                    X_mesh,Y_mesh = np.zeros((Imax-Imin+1, Jmax-Jmin+1)),np.zeros((Imax-Imin+1, Jmax-Jmin+1))
                    for line in range((Imax-Imin+1)*(Jmax-Jmin+1)*Num_T):
                        i,j,k,t = (float(t) for t in re.split(',| |\[|\]|\(|\)',
                                                              fp.readline()[:-1])
                                   if len(t)>0)
                        Data[int(i),int(j),int(k)] = t
                        #TODO一个问题是这个不是中心点，第二个是似乎x和y搞反了
                        X_mesh[int(i),int(j)],Y_mesh[int(i),int(j)] = (Imin+i+1/2)/(Imax-Imin+1), (Jmin+j+1/2)/(Jmax-Jmin+1)
                    # x=np.linspace((xh - xl)/(x2 - x1 +1)*(0.5),1-(xh - xl)/(x2 - x1 +1)*(0.5),nx)
                    # y=np.linspace((yh - yl)/(y2 - y1 +1)*(0.5),1-(yh - yl)/(y2 - y1 +1)*(0.5),ny)
                    # X_mesh,Y_mesh = np.meshgrid(x,y)
                    # 光子电子离子  ->  电子离子光子
                    Data = Data[:,:,(1,2,0)]
                
                X_2d3t_t005 = np.concatenate([X_mesh.reshape(-1,1),Y_mesh.reshape(-1,1),1.0*np.ones(((Imax-Imin+1)*(Jmax-Jmin+1),1))],axis=1)
                upred_t005 = self.predict(X_2d3t_t005).reshape(-1,1)
                upred_t005np = upred_t005.detach().numpy()
                T_mesht005 = upred_t005np.reshape(Data.shape)
                
                #print Relative L2 error
                
                
                #画参考解和误差
                maxve = np.max(Data[:,:,0])
                minve = np.min(Data[:,:,0])
                plt.figure(figsize=(10,8),dpi=100)
                plt.subplot(3,3,0+1)
                plt.imshow(Data[:,:,0].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower',vmin=minve,vmax=maxve)
                # te=plt.imshow(Data[:,:,0].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
                # te.set_clim(minve,maxve)
                cbarte=plt.colorbar()
                cbarte.set_ticks(list(np.linspace(minve, maxve, 5)))
                # plt.contourf(X_mesh,Y_mesh,Data[:,:,0],levels=50,cmap='jet',vmin=minve,vmax=maxve)
                # cbar=plt.colorbar()
                # cbar.ax.yaxis.set_major_formatter(ScalarFormatter()) 
                # cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
                # plt.clim(minve,maxve)
                plt.xlabel("$x$", fontsize=12)
                plt.ylabel("$y$", fontsize=12)
                plt.title("True solution Te")
                maxvi = np.max(Data[:,:,1])
                minvi = np.min(Data[:,:,1])
                plt.subplot(3,3,1+1)
                plt.imshow(Data[:,:,1].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower',vmin=minvi,vmax=maxvi)
                # ti=plt.imshow(Data[:,:,1].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
                # ti.set_clim(minvi,maxvi)
                cbarti=plt.colorbar()
                cbarti.set_ticks(list(np.linspace(minvi, maxvi, 5)))
                # plt.contourf(X_mesh,Y_mesh,Data[:,:,1],levels=50,cmap='jet',vmin=minvi,vmax=maxvi)
                # cbar=plt.colorbar()
                # cbar.ax.yaxis.set_major_formatter(ScalarFormatter()) 
                # cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
                # plt.clim(minvi,maxvi)
                plt.xlabel("$x$", fontsize=12)
                plt.ylabel("$y$", fontsize=12)
                plt.title("True solution Ti")
                maxvr = np.max(Data[:,:,2])
                minvr = np.min(Data[:,:,2])
                plt.subplot(3,3,2+1)
                plt.imshow(Data[:,:,2].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower',vmin=minvr,vmax=maxvr)
                # tr=plt.imshow(Data[:,:,2].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
                # tr.set_clim(minvr,maxvr)
                cbartr=plt.colorbar()
                cbartr.set_ticks(list(np.linspace(minvr, maxvr, 5)))
                # plt.contourf(X_mesh,Y_mesh,Data[:,:,2],levels=50,cmap='jet',vmin=minvr,vmax=maxvr)
                # cbar=plt.colorbar()
                # cbar.ax.yaxis.set_major_formatter(ScalarFormatter()) 
                # cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
                # plt.clim(minvr,maxvr)
                plt.xlabel("$x$", fontsize=12)
                plt.ylabel("$y$", fontsize=12)
                plt.title("True solution Tr")
                
                
                plt.subplot(3,3,0+1+3)
                plt.imshow(T_mesht005[:,:,0].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower',vmin=minve,vmax=maxve)
                # plt.imshow(T_mesht005[:,:,0].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
                # tep=plt.imshow(T_mesht005[:,:,0].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
                # tep.set_clim(minve,maxve)
                cbartep=plt.colorbar()
                cbartep.set_ticks(list(np.linspace(minve, maxve, 5)))
                # plt.contourf(X_mesh,Y_mesh,T_mesht005[:,:,0],levels=50,cmap='jet',vmin=minve,vmax=maxve)
                # plt.clim(np.min(T_mesh[:,:,0].T),np.max(T_mesh[:,:,0].T))
                # cbar=plt.colorbar()
                # cbar.ax.yaxis.set_major_formatter(ScalarFormatter()) 
                # cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
                # plt.clim(minve,maxve)
                plt.xlabel("$x$", fontsize=12)
                plt.ylabel("$y$", fontsize=12)
                plt.title("PINN Te")
                plt.subplot(3,3,1+1+3)
                plt.imshow(T_mesht005[:,:,1].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower',vmin=minvi,vmax=maxvi)
                # plt.imshow(T_mesht005[:,:,1].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
                # tip=plt.imshow(T_mesht005[:,:,1].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
                # tip.set_clim(minvi,maxvi)
                cbartip=plt.colorbar()
                cbartip.set_ticks(list(np.linspace(minvi, maxvi, 5)))
                # plt.contourf(X_mesh,Y_mesh,T_mesht005[:,:,1],levels=50,cmap='jet',vmin=minvi,vmax=maxvi)
                # plt.clim(np.min(T_mesh[:,:,1].T),np.max(T_mesh[:,:,1].T))
                # cbar=plt.colorbar()
                # cbar.ax.yaxis.set_major_formatter(ScalarFormatter()) 
                # cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
                # plt.clim(minvi,maxvi)
                plt.xlabel("$x$", fontsize=12)
                plt.ylabel("$y$", fontsize=12)
                plt.title("PINN Ti")
                plt.subplot(3,3,2+1+3)
                plt.imshow(T_mesht005[:,:,2].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower',vmin=minvr,vmax=maxvr)
                # plt.imshow(T_mesht005[:,:,2].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
                # trp=plt.imshow(T_mesht005[:,:,2].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
                # trp.set_clim(minvr,maxvr)
                cbartrp=plt.colorbar()
                cbartrp.set_ticks(list(np.linspace(minvr, maxvr, 5)))
                # plt.contourf(X_mesh,Y_mesh,T_mesht005[:,:,2],levels=50,cmap='jet',vmin=minvr,vmax=maxvr)
                # plt.clim(np.min(T_mesh[:,:,2]),np.max(T_mesh[:,:,2]))
                # cbar=plt.colorbar()
                # cbar.ax.yaxis.set_major_formatter(ScalarFormatter()) 
                # cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
                # plt.clim(minvr,maxvr)
                plt.xlabel("$x$", fontsize=12)
                plt.ylabel("$y$", fontsize=12)
                plt.title("PINN Tr")
                
                plt.subplot(3,3,0+1+6)
                plt.imshow((np.abs(Data[:,:,0]-T_mesht005[:,:,0])).T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
                plt.colorbar()
                # cbarteer=plt.colorbar()
                # cbarteer.set_ticks(list(np.linspace(1/100, 1/1000, 10)))
                # plt.contourf(X_mesh,Y_mesh,np.abs(Data[:,:,0]-T_mesht005[:,:,0]),levels=50,cmap='jet')
                # cbar=plt.colorbar()
                # cbar.ax.yaxis.set_major_formatter(ScalarFormatter()) 
                # cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
                plt.xlabel("$x$", fontsize=12)
                plt.ylabel("$y$", fontsize=12)
                plt.title("point error Te")
                plt.subplot(3,3,1+1+6)
                plt.imshow((np.abs(Data[:,:,1]-T_mesht005[:,:,1])).T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
                plt.colorbar()
                # cbartier=plt.colorbar()
                # cbartier.set_ticks(list(np.linspace(1/100, 1/1000, 10)))
                # plt.contourf(X_mesh,Y_mesh,np.abs(Data[:,:,1]-T_mesht005[:,:,1]),levels=50,cmap='jet')
                # cbar=plt.colorbar()
                # cbar.ax.yaxis.set_major_formatter(ScalarFormatter()) 
                # cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
                plt.xlabel("$x$", fontsize=12)
                plt.ylabel("$y$", fontsize=12)
                plt.title("point error Ti")
                plt.subplot(3,3,2+1+6)
                plt.imshow((np.abs(Data[:,:,2]-T_mesht005[:,:,2])).T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
                plt.colorbar()
                # cbartrer=plt.colorbar()
                # cbartrer.set_ticks(list(np.linspace(1/100, 1/1000, 10)))
                # plt.contourf(X_mesh,Y_mesh,np.abs(Data[:,:,2]-T_mesht005[:,:,2]),levels=50,cmap='jet')
                # cbar=plt.colorbar()
                # cbar.ax.yaxis.set_major_formatter(ScalarFormatter()) 
                # cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
                plt.xlabel("$x$", fontsize=12)
                plt.ylabel("$y$", fontsize=12)
                plt.title("point error Tr")
                plt.suptitle('t=1.0')
                
                try:
                    plt.tight_layout()
                    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, \
                        wspace=None, hspace=None)
                    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, \
                    #     wspace=None, hspace=None)
                    
                    # plt.subplots(3,3,constrained_layout=True)#改成True后可以自动调整绘图区域在整个figure上的位置
                    # plt.tight_layout(pad=1,h_pad=5.0,w_pad=5.0) 
                    plt.savefig('./figures/'+FolderName+'3x3.png',bbox_inches='tight')
                    # print('hello')
                except:
                    print('save img failed')
                    plt.show()
                plt.close()  # 这是关闭绘图区，自己查看我打算画在一张图上了
            """    
            self.net_u.zero_grad() # model.zero_grad 与 opt.zero_grad 完全等价，opt甚至是调用了model.
            loss.backward()
            return loss
            
            
        # ============================================================
        # ****  Adam  
        # ============================================================ 

        for it in range(nIter):
            
            #opt.zero_grad()
            #loss.backward()
            opt.step(lambda : complus_loss(it))  # LBFGS特有的使用方式（因为需要多次计算Loss）对一阶算法也适用
            scheduler.step()

        # ============================================================
        # ****  LBFGS , LBFGS 可以自行获取迭代次数 用以判断输出，无需传入
        # ============================================================ 
        opt_lbfgs.step(complus_loss)
            
            
            
        return lr, Trboundloss, Teiniloss, Tiiniloss, Triniloss, TeL2, TiL2, TrL2, Temseloss, Timseloss, Trmseloss, MSE_history, a_history, L2error_u, L1error_u, Linferror_u


    def predict(self, X_star):
        temp_x = torch.from_numpy(X_star[:, 0:1]).to(**DeviceDtype)
        temp_y = torch.from_numpy(X_star[:, 1:2]).to(**DeviceDtype)
        temp_t = torch.from_numpy(X_star[:, 2:3]).to(**DeviceDtype)
        u_star = self.net(torch.cat((temp_x, temp_y, temp_t), 1))
        return u_star.cpu().data
