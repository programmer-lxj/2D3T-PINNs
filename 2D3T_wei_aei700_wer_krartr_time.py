#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022.11.10
author : lei xiaojun

"""

import sys,os
sys.path.insert(0, '../Utilities/')

import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from matplotlib.ticker import MaxNLocator
from sub_2D3T_wei_aei700_wer_krartr_time import PhysicsInformedNN 
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import time
import matplotlib.gridspec as gridspec



#种子要改回去!!!
np.random.seed(12)
torch.manual_seed(12)
torch.cuda.manual_seed_all(12)
torch.backends.cudnn.deterministic = True

# 保存文件的子文件夹，用于多次运行不覆盖FolderName = 'Train_6000_30000_09061400/'
FolderName = 'Train_6000_30000_2406032237/'

try:
    os.mkdir('./figures/')
    print('Create Folder: ./figures/')
except:
    pass
try:
    os.mkdir('./figures/'+FolderName)
    print('Create Folder: ./figures/'+FolderName)
except:
    pass

#得先运行一次这个程序才能找到数据文件
if __name__ == "__main__": 
    if torch.cuda.is_available():
        device = 'cuda'
        torch.backends.cudnn.enabled =True
        torch.backends.cudnn.benchmark = True
        # 下面2个选项，针对NVIDIA 30系 Ampere架构 TensorCore优化
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else: 
        device = 'cpu'
    print('Using GPU, %s' % torch.cuda.get_device_name(0)) if device=='cuda' else   print('Using CPU')
    #这是CH的物理参数
    gamme_e_ch=45
    gamme_i_ch=70
    gamme_r_ch=0.007568
    rou = 1.1
    Ae = 81
    Ai = 0.02
    Ar = 2.1e2/(rou*rou)
    ce = 1.5*gamme_e_ch
    ci = 1.5*gamme_i_ch
    cr = (1/4)*gamme_r_ch
    beta = 3
    
    Aei = 700
    # Aei = 400
    Aer = 79
    
    # gamme_e_ch=1
    # gamme_i_ch=1
    # gamme_r_ch=1
    # rou = 1
    # Ae = 1
    # Ai = 1
    # Ar = 1
    # ce = 1.5*gamme_e_ch
    # ci = 1.5*gamme_i_ch
    # cr = (1/4)*gamme_r_ch
    # beta = 1
    
    # Aei = 1
    # Aer = 1
    
    #这是SiO2的物理参数
    # rou = 2.5
    # Ae = 60
    # Ai = 0.00017
    # Ar = 9e2/(rou**1.5)
    # ce = 40
    # ci = 40
    # cr = 0.007568
    # beta = 2.4
    
    # Aei = 4000
    # Aer = 140
    
    
    #Max_iter = 20001    #15001
    # Max_iter = 50000    #15001
    # Max_iter = 5001    #15001
    Max_iter = 6001    #15001
    #Max_iter = 21    #15001
    
    
    #极坐标下半径和角度  x-->r y-->theta
    x_interface = np.array([0, 1])
    y_interface = np.array([0, 1])
    N_x = len(x_interface) - 1
    N_y = len(y_interface) - 1
    
    xlbse = x_interface[0]
    xubse = x_interface[-1]
    ylbse = y_interface[0]
    yubse = y_interface[-1]
    
    Num_interface = (N_x-1)
    Num_subdomain = N_x
    NN_depth = [6]
    NN_width = [60]
    NN_layers_total = []
    #残差 电子*10 离子*100 
    #点数，网络加深
    
    for sd in range(Num_subdomain):
        NN_layer_sd = [3] + [NN_width[sd]] * NN_depth[sd] + [3]
        NN_layers_total.append(NN_layer_sd)
    

    N_u_subdomain_total = Num_subdomain * [60]
    N_f_interface       = 80
    N_f_interface_total = Num_interface * [N_f_interface]   # [99, 99, 99], 99 points is taken at each interface
    N_f                 = 100
    N_test              = 500
    N_f_total = [4000]
    N_test_total           = Num_subdomain * [N_test]
    
    

    # lb = np.array([0, 0, 0])
    # ub = np.array([1, 1, 1])
    
    lb = np.array([0, 0, 0])
    ub = np.array([1, 1, 1.1])

    
    
    Nb = 1200
    N0 = 1200



    #############串行PINN随机采点
    #初边值点
    Total_Num = 10000
    x_star = np.random.rand(Total_Num, 1) * (ub[0]-lb[0]) + lb[0]
    y_star = np.random.rand(Total_Num, 1) * (ub[1]-lb[1]) + lb[1]
    t_star = np.random.rand(Total_Num, 1) * (ub[2]-lb[2]) + lb[2]
    Nbse = 1200
    N0se = 1200
    
    
    #边界条件，只采四条边坐标，不用传精确解，反正都是常数，初始条件也是
    idx1 = np.random.choice(Total_Num, Nbse, replace=False)
    X_xlb = np.full((Nbse,1), lb[0])
    Y_xlb = y_star[idx1,:]
    T_xlb = t_star[idx1,:]
    Xlbtrain = np.concatenate((X_xlb,Y_xlb,T_xlb),axis=1)

    idx2 = np.random.choice(Total_Num, Nbse, replace=False)
    X_xub = np.full((Nbse,1), ub[0])
    Y_xub = y_star[idx2,:]
    T_xub = t_star[idx2,:]
    Xubtrain = np.concatenate((X_xub,Y_xub,T_xub),axis=1)

    idx3 = np.random.choice(Total_Num, Nbse, replace=False)
    X_ylb = x_star[idx3,:]
    Y_ylb = np.full((Nbse,1), lb[1])
    T_ylb = t_star[idx3,:]
    Ylbtrain = np.concatenate((X_ylb,Y_ylb,T_ylb),axis=1)

    idx4 = np.random.choice(Total_Num, Nbse, replace=False)
    X_yub = x_star[idx4,:]
    Y_yub = np.full((Nbse,1), ub[1])
    T_yub = t_star[idx4,:]
    Yubtrain = np.concatenate((X_yub,Y_yub,T_yub),axis=1)
    # print(T_yub)
    
    #初始条件
    idx5 = np.random.choice(Total_Num, N0, replace=False)
    X0 = x_star[idx5,:]
    Y0 = y_star[idx5,:]
    T0 = np.full((N0,1), lb[2])
    T0train = np.concatenate((X0,Y0,T0),axis=1)
    
    

    #  自适应采残差点
    X_f_train_total = lb+(ub-lb)*lhs(3,N_f_total[0])

    #监督点
    X_super = lb+(ub-lb)*lhs(3,100)
    x=X_super[:,0:1] 
    y=X_super[:,1:2]
    t=X_super[:,2:3]

    

    # ******************************************************PINN
    

    # 这一堆   只能拿来 随机摇奖  没法画图（顺序不是网格，即使reshape）
    X_test=lb+(ub-lb)*lhs(3,625)
    # X_star=X_f_train_total
    X_star=X_test
    x=X_star[:,0:1] 
    y=X_star[:,1:2]
    t=X_star[:,2:3]
    t0_1=np.full((625,1), 0.1)
    X_plot= np.concatenate((x,y,t0_1),axis=1)
    #应该是竖着摆，不是横着摆吧？
    # u_star = np.concatenate((u_starte,u_starti,u_startr),axis=1)
    #TODO从这里读取参考解
    
    
    # #为解决自恰，从1e-8开始模拟，从文件读1e-8的初值
    # with open('./sol0.6656063988.txt','r') as fp:
    #     import numpy as np
    #     from itertools import product
    #     import re
    #     DIM, Num_T, _ = eval(fp.readline())
    #     print(DIM, Num_T, _)
    #     (Imin,Jmin),(Imax,Jmax),_ = eval(fp.readline())
    #     print((Imin,Jmin),(Imax,Jmax),_)
    #     Data = np.zeros((Imax-Imin+1, Jmax-Jmin+1, Num_T))
    #     X_mesh,Y_mesh = np.zeros((Imax-Imin+1, Jmax-Jmin+1)),np.zeros((Imax-Imin+1, Jmax-Jmin+1))
    #     for line in range((Imax-Imin+1)*(Jmax-Jmin+1)*Num_T):
    #         i,j,k,t = (float(t) for t in re.split(',| |\[|\]|\(|\)',
    #                                               fp.readline()[:-1])
    #                     if len(t)>0)
    #         Data[int(i),int(j),int(k)] = t
    #         X_mesh[int(i),int(j)],Y_mesh[int(i),int(j)] = (Imin+i+1/2)/(Imax-Imin+1), (Jmin+j+1/2)/(Jmax-Jmin+1)
    #     # x=np.linspace((xh - xl)/(x2 - x1 +1)*(0.5),1-(xh - xl)/(x2 - x1 +1)*(0.5),nx)
    #     # y=np.linspace((yh - yl)/(y2 - y1 +1)*(0.5),1-(yh - yl)/(y2 - y1 +1)*(0.5),ny)
    #     # X_mesh,Y_mesh = np.meshgrid(x,y)
    #     # 光子电子离子  ->  电子离子光子
    #     Data = Data[:,:,(1,2,0)]
        # u0 = Data.reshape((Imax-Imin+1)*(Jmax-Jmin+1),Num_T)
        # T0train = np.concatenate([X_mesh.reshape(-1,1),Y_mesh.reshape(-1,1),0*np.ones(((Imax-Imin+1)*(Jmax-Jmin+1),1))],axis=1)
    
    # model = PhysicsInformedNN(Xlbtrain, Xubtrain, Ylbtrain, Yubtrain, T0train, u0, X_f_train_total, NN_layers_total,\
    #                           rou, ce, ci, cr, beta, Ae, Ai, Ar, Aei, Aer)

    # model = PhysicsInformedNN(Xlbtrain, Xubtrain, Ylbtrain, Yubtrain, T0train, X_f_train_total, NN_layers_total,\
    #                           rou, ce, ci, cr, beta, Ae, Ai, Ar, Aei, Aer)

    # start_time = time.time()                
    # # Teiniloss, Tiiniloss, Triniloss, TeL2, TiL2, TrL2, Teloss, Tiloss, Trloss, MSE_hist, a_hist, L2error_u, L1error_u, Linferror_u = model.train(Max_iter,X_star,u_star)
    # Trboundloss, Teiniloss, Tiiniloss, Triniloss, TeL2, TiL2, TrL2, Teloss, Tiloss, Trloss, MSE_hist, a_hist, L2error_u, L1error_u, Linferror_u = model.train(Max_iter,X_star)
    # elapsed = time.time() - start_time                
    # print('Training time: %.4f' % (elapsed))
   
    train_time=0
    Aei = False
    # Aei  : old
    # Aei2 : new
    for Aei2 in [70,400,700]:  
    #for Aei2 in [700]:  
    # for Aei2 in [70,7000]:
        
        print('Aei: %d' % (Aei))
        #model = PhysicsInformedNN(Xlbtrain, Xubtrain, Ylbtrain, Yubtrain, T0train, X_f_train_total, NN_layers_total,\
        #                          rou, ce, ci, cr, beta, Ae, Ai, Ar, Aei2, Aer, lrin=1e-3 if not Aei else 2e-5)
        model = PhysicsInformedNN(Xlbtrain, Xubtrain, Ylbtrain, Yubtrain, T0train, X_f_train_total, NN_layers_total,\
                                  rou, ce, ci, cr, beta, Ae, Ai, Ar, Aei2, Aer, lrin=1e-3 if not Aei else 1e-4)
        # 第一次 Aei False，不处理；后面才读 Aei
        if (Aei):
            model.net_u.load_state_dict(torch.load('./figures/'+FolderName+'_%d_after.pt'%(Aei)))
        start_time = time.time()                
        # Teiniloss, Tiiniloss, Triniloss, TeL2, TiL2, TrL2, Teloss, Tiloss, Trloss, MSE_hist, a_hist, L2error_u, L1error_u, Linferror_u = model.train(Max_iter,X_star,u_star)
        lr, Trboundloss, Teiniloss, Tiiniloss, Triniloss, TeL2, TiL2, TrL2, Teloss, Tiloss, Trloss, MSE_hist, a_hist, L2error_u, L1error_u, Linferror_u = model.train(Max_iter,X_star)
        elapsed = time.time() - start_time
        train_time += elapsed   
        torch.save(model.net_u.state_dict(), './figures/'+FolderName+'_%d_after.pt'%(Aei2))   
        Aei = Aei2
        print('Training time: %.4f' % (elapsed))
    print('Total Training time: %.4f' % (train_time))
    """
    #for Aei2 in [70,400]:  
    #for Aei2 in [400,7000]:
    for Aei2 in [70,400,700]:
    #for Aei2 in [70,450,7000]:  
        Aei = Aei2
        print('Aei: %d' % (Aei))
        model = PhysicsInformedNN(Xlbtrain, Xubtrain, Ylbtrain, Yubtrain, T0train, X_f_train_total, NN_layers_total,\
                                  rou, ce, ci, cr, beta, Ae, Ai, Ar, Aei, Aer)
        start_time = time.time()                
        # Teiniloss, Tiiniloss, Triniloss, TeL2, TiL2, TrL2, Teloss, Tiloss, Trloss, MSE_hist, a_hist, L2error_u, L1error_u, Linferror_u = model.train(Max_iter,X_star,u_star)
        lr, Trboundloss, Teiniloss, Tiiniloss, Triniloss, TeL2, TiL2, TrL2, Teloss, Tiloss, Trloss, MSE_hist, a_hist, L2error_u, L1error_u, Linferror_u = model.train(Max_iter,X_star)
        elapsed = time.time() - start_time                
        print('Training time: %.4f' % (elapsed))
    """    
    """ 
    try:
        os.mkdir('./figures/')
        print('Create Folder: ./figures/')
    except:
        pass
    try:
        os.mkdir('./figures/'+FolderName)
        print('Create Folder: ./figures/'+FolderName)
    except:
        pass
    """
    
    t=np.linspace(0, 0.1, 100)
    X_tl = np.concatenate([0.1*np.ones((100,1)).reshape(-1,1),0.5*np.ones((100,1)).reshape(-1,1),t.reshape(-1, 1)],axis=1)
    upredtl = model.predict(X_tl)
    upredtlnp = upredtl.detach().numpy()
    
    X_tr = np.concatenate([0.9*np.ones((100,1)).reshape(-1,1),0.5*np.ones((100,1)).reshape(-1,1),t.reshape(-1, 1)],axis=1)
    upredtr = model.predict(X_tr)
    upredtrnp = upredtr.detach().numpy()
    
    X_tlow = np.concatenate([0.5*np.ones((100,1)).reshape(-1,1),0.1*np.ones((100,1)).reshape(-1,1),t.reshape(-1, 1)],axis=1)
    upredtlow = model.predict(X_tlow)
    upredtlownp = upredtlow.detach().numpy()
    
    X_tu = np.concatenate([0.5*np.ones((100,1)).reshape(-1,1),0.9*np.ones((100,1)).reshape(-1,1),t.reshape(-1, 1)],axis=1)
    upredtu = model.predict(X_tu)
    upredtunp = upredtu.detach().numpy()
    
    X_tm = np.concatenate([0.5*np.ones((100,1)).reshape(-1,1),0.5*np.ones((100,1)).reshape(-1,1),t.reshape(-1, 1)],axis=1)
    upredtm = model.predict(X_tm)
    upredtmnp = upredtm.detach().numpy()
    
    X_ty1 = np.concatenate([0.5*np.ones((100,1)).reshape(-1,1),1*np.ones((100,1)).reshape(-1,1),t.reshape(-1, 1)],axis=1)
    upredty1 = model.predict(X_ty1)
    upredty1np = upredty1.detach().numpy()
    t=np.squeeze(t) 
    #温度随时间曲线
    plt.figure(figsize=(18,16),dpi=100)  
    plt.subplot(2,3,1)
    plt.plot(t, upredtlnp[:,0:1],  'r.-', linewidth = 1,  label = 'Te') 
    plt.plot(t, upredtlnp[:,1:2],  'g--', linewidth = 1,  label = 'Ti')  
    plt.plot(t, upredtlnp[:,2:3],  'b-', linewidth = 1,  label = 'Tr')   
    plt.tick_params(labelsize=16)      #坐标轴字体大小
    plt.xlabel('t', fontsize = 16)
    plt.ylabel('Temperature', fontsize = 16)
    plt.title('Temperature0.1_0.5', fontsize = 16)
    # plt.legend(loc='upper right') 
    plt.legend(prop = {'size':16})   #图例字体大小
    
    plt.subplot(2,3,2)
    plt.plot(t, upredtrnp[:,0:1],  'r.-', linewidth = 1,  label = 'Te') 
    plt.plot(t, upredtrnp[:,1:2],  'g--', linewidth = 1,  label = 'Ti')  
    plt.plot(t, upredtrnp[:,2:3],  'b-', linewidth = 1,  label = 'Tr') 
    plt.tick_params(labelsize=16)      #坐标轴字体大小       
    plt.xlabel('t', fontsize = 16)
    plt.ylabel('Temperature', fontsize = 16)
    plt.title('Temperature0.9_0.5', fontsize = 16)
    # plt.legend(loc='upper right') 
    plt.legend(prop = {'size':16})   #图例字体大小
    
    plt.subplot(2,3,3)
    plt.plot(t, upredtlownp[:,0:1],  'r.-', linewidth = 1,  label = 'Te') 
    plt.plot(t, upredtlownp[:,1:2],  'g--', linewidth = 1,  label = 'Ti')  
    plt.plot(t, upredtlownp[:,2:3],  'b-', linewidth = 1,  label = 'Tr') 
    plt.tick_params(labelsize=16)      #坐标轴字体大小         
    plt.xlabel('t', fontsize = 16)
    plt.ylabel('Temperature', fontsize = 16)
    plt.title('Temperature0.5_0.1', fontsize = 16)
    # plt.legend(loc='upper right') 
    plt.legend(prop = {'size':16})   #图例字体大小
    
    plt.subplot(2,3,4)
    plt.plot(t, upredtunp[:,0:1],  'r.-', linewidth = 1,  label = 'Te') 
    plt.plot(t, upredtunp[:,1:2],  'g--', linewidth = 1,  label = 'Ti')  
    plt.plot(t, upredtunp[:,2:3],  'b-', linewidth = 1,  label = 'Tr')   
    plt.tick_params(labelsize=16)      #坐标轴字体大小       
    plt.xlabel('t', fontsize = 16)
    plt.ylabel('Temperature', fontsize = 16)
    plt.title('Temperature0.5_0.9', fontsize = 16)
    # plt.legend(loc='upper right') 
    plt.legend(prop = {'size':16})   #图例字体大小
    
    plt.subplot(2,3,5)
    plt.plot(t, upredtmnp[:,0:1],  'r.-', linewidth = 1,  label = 'Te') 
    plt.plot(t, upredtmnp[:,1:2],  'g--', linewidth = 1,  label = 'Ti')  
    plt.plot(t, upredtmnp[:,2:3],  'b-', linewidth = 1,  label = 'Tr')  
    plt.tick_params(labelsize=16)      #坐标轴字体大小        
    plt.xlabel('t', fontsize = 16)
    plt.ylabel('Temperature', fontsize = 16)
    plt.title('Temperature0.5_0.5', fontsize = 16)
    # plt.legend(loc='upper right') 
    plt.legend(prop = {'size':16})   #图例字体大小
    
    plt.subplot(2,3,6)
    plt.plot(t, upredty1np[:,0:1],  'r.-', linewidth = 1,  label = 'Te') 
    plt.plot(t, upredty1np[:,1:2],  'g--', linewidth = 1,  label = 'Ti')  
    plt.plot(t, upredty1np[:,2:3],  'b-', linewidth = 1,  label = 'Tr')   
    plt.tick_params(labelsize=16)      #坐标轴字体大小       
    plt.xlabel('t', fontsize = 16)
    plt.ylabel('Temperature', fontsize = 16)
    plt.title('Temperature0.5_1', fontsize = 16)
    # plt.legend(loc='upper right') 
    plt.legend(prop = {'size':16})   #图例字体大小
    try:
        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, \
            wspace=0.5, hspace=None)
        plt.savefig('./figures/'+FolderName+'2x3Temperature.png',bbox_inches='tight')
    except:
        print('save img failed')
        plt.show()
    plt.close()  # 这是关闭绘图区，自己查看我打算画在一张图上了
    
    
    # #PINN给2d3t初始值
    x1 = 0
    x2 = 31
    y1 = 0
    y2 = 31
    xl = 0
    xh = 1
    yl = 0
    yh = 1

    nx = x2 - x1 + 1
    ny = y2 - y1 + 1
    
    
    #计算5个时刻的三种范数意义下的误差
    with open('./sol1_wei_aei700_wer_krartr_80_1.txt','r') as fp:
    # with open('./sol1e-8.txt','r') as fp:
        import numpy as np
        from itertools import product
        import re
        DIM, Num_T, _ = eval(fp.readline())
        print(DIM, Num_T, _)
        (Imin,Jmin),(Imax,Jmax),_ = eval(fp.readline())
        print((Imin,Jmin),(Imax,Jmax),_)
        Data1efu5 = np.zeros((Imax-Imin+1, Jmax-Jmin+1, Num_T))
        X_mesh,Y_mesh = np.zeros((Imax-Imin+1, Jmax-Jmin+1)),np.zeros((Imax-Imin+1, Jmax-Jmin+1))
        for line in range((Imax-Imin+1)*(Jmax-Jmin+1)*Num_T):
            i,j,k,t = (float(t) for t in re.split(',| |\[|\]|\(|\)',
                                                  fp.readline()[:-1])
                       if len(t)>0)
            Data1efu5[int(i),int(j),int(k)] = t
            #TODO一个问题是这个不是中心点，第二个是似乎x和y搞反了
            X_mesh[int(i),int(j)],Y_mesh[int(i),int(j)] = (Imin+i+1/2)/(Imax-Imin+1), (Jmin+j+1/2)/(Jmax-Jmin+1)
        # x=np.linspace((xh - xl)/(x2 - x1 +1)*(0.5),1-(xh - xl)/(x2 - x1 +1)*(0.5),nx)
        # y=np.linspace((yh - yl)/(y2 - y1 +1)*(0.5),1-(yh - yl)/(y2 - y1 +1)*(0.5),ny)
        # X_mesh,Y_mesh = np.meshgrid(x,y)
        # 光子电子离子  ->  电子离子光子
        Data1efu5 = Data1efu5[:,:,(1,2,0)].reshape(-1,3)
    
    X_2d3t_1efu5 = np.concatenate([X_mesh.reshape(-1,1),Y_mesh.reshape(-1,1),(1e-5)*np.ones(((Imax-Imin+1)*(Jmax-Jmin+1),1))],axis=1)
    upred_1efu5 = model.predict(X_2d3t_1efu5).reshape(-1,3)
    #np.savetxt('pinnxp%d.txt' % (model.p),X_2d3t)
    upred_1efu5 = upred_1efu5.detach().numpy()
    
    
    #计算5个时刻的三种范数意义下的误差
    with open('./sol1_wei_aei700_wer_krartr_80_1.txt','r') as fp:
    # with open('./sol1e-8.txt','r') as fp:
        import numpy as np
        from itertools import product
        import re
        DIM, Num_T, _ = eval(fp.readline())
        print(DIM, Num_T, _)
        (Imin,Jmin),(Imax,Jmax),_ = eval(fp.readline())
        print((Imin,Jmin),(Imax,Jmax),_)
        Data0p3 = np.zeros((Imax-Imin+1, Jmax-Jmin+1, Num_T))
        X_mesh,Y_mesh = np.zeros((Imax-Imin+1, Jmax-Jmin+1)),np.zeros((Imax-Imin+1, Jmax-Jmin+1))
        for line in range((Imax-Imin+1)*(Jmax-Jmin+1)*Num_T):
            i,j,k,t = (float(t) for t in re.split(',| |\[|\]|\(|\)',
                                                  fp.readline()[:-1])
                       if len(t)>0)
            Data0p3[int(i),int(j),int(k)] = t
            #TODO一个问题是这个不是中心点，第二个是似乎x和y搞反了
            X_mesh[int(i),int(j)],Y_mesh[int(i),int(j)] = (Imin+i+1/2)/(Imax-Imin+1), (Jmin+j+1/2)/(Jmax-Jmin+1)
        # x=np.linspace((xh - xl)/(x2 - x1 +1)*(0.5),1-(xh - xl)/(x2 - x1 +1)*(0.5),nx)
        # y=np.linspace((yh - yl)/(y2 - y1 +1)*(0.5),1-(yh - yl)/(y2 - y1 +1)*(0.5),ny)
        # X_mesh,Y_mesh = np.meshgrid(x,y)
        # 光子电子离子  ->  电子离子光子
        Data0p3 = Data0p3[:,:,(1,2,0)].reshape(-1,3)
    
    X_2d3t_0p3 = np.concatenate([X_mesh.reshape(-1,1),Y_mesh.reshape(-1,1),(0.3)*np.ones(((Imax-Imin+1)*(Jmax-Jmin+1),1))],axis=1)
    upred_0p3 = model.predict(X_2d3t_0p3).reshape(-1,3)
    #np.savetxt('pinnxp%d.txt' % (model.p),X_2d3t)
    upred_0p3 = upred_0p3.detach().numpy()
    
    #计算5个时刻的三种范数意义下的误差
    with open('./sol1_wei_aei700_wer_krartr_80_1.txt','r') as fp:
    # with open('./sol1e-8.txt','r') as fp:
        import numpy as np
        from itertools import product
        import re
        DIM, Num_T, _ = eval(fp.readline())
        print(DIM, Num_T, _)
        (Imin,Jmin),(Imax,Jmax),_ = eval(fp.readline())
        print((Imin,Jmin),(Imax,Jmax),_)
        Data0p5 = np.zeros((Imax-Imin+1, Jmax-Jmin+1, Num_T))
        X_mesh,Y_mesh = np.zeros((Imax-Imin+1, Jmax-Jmin+1)),np.zeros((Imax-Imin+1, Jmax-Jmin+1))
        for line in range((Imax-Imin+1)*(Jmax-Jmin+1)*Num_T):
            i,j,k,t = (float(t) for t in re.split(',| |\[|\]|\(|\)',
                                                  fp.readline()[:-1])
                       if len(t)>0)
            Data0p5[int(i),int(j),int(k)] = t
            #TODO一个问题是这个不是中心点，第二个是似乎x和y搞反了
            X_mesh[int(i),int(j)],Y_mesh[int(i),int(j)] = (Imin+i+1/2)/(Imax-Imin+1), (Jmin+j+1/2)/(Jmax-Jmin+1)
        # x=np.linspace((xh - xl)/(x2 - x1 +1)*(0.5),1-(xh - xl)/(x2 - x1 +1)*(0.5),nx)
        # y=np.linspace((yh - yl)/(y2 - y1 +1)*(0.5),1-(yh - yl)/(y2 - y1 +1)*(0.5),ny)
        # X_mesh,Y_mesh = np.meshgrid(x,y)
        # 光子电子离子  ->  电子离子光子
        Data0p5 = Data0p5[:,:,(1,2,0)].reshape(-1,3)
        
    X_2d3t_0p5 = np.concatenate([X_mesh.reshape(-1,1),Y_mesh.reshape(-1,1),(0.5)*np.ones(((Imax-Imin+1)*(Jmax-Jmin+1),1))],axis=1)
    upred_0p5 = model.predict(X_2d3t_0p5).reshape(-1,3)
    #np.savetxt('pinnxp%d.txt' % (model.p),X_2d3t)
    upred_0p5 = upred_0p5.detach().numpy()
        
    
    #计算5个时刻的三种范数意义下的误差
    with open('./sol1_wei_aei700_wer_krartr_80_1.txt','r') as fp:
    # with open('./sol1e-8.txt','r') as fp:
        import numpy as np
        from itertools import product
        import re
        DIM, Num_T, _ = eval(fp.readline())
        print(DIM, Num_T, _)
        (Imin,Jmin),(Imax,Jmax),_ = eval(fp.readline())
        print((Imin,Jmin),(Imax,Jmax),_)
        Data0p7 = np.zeros((Imax-Imin+1, Jmax-Jmin+1, Num_T))
        X_mesh,Y_mesh = np.zeros((Imax-Imin+1, Jmax-Jmin+1)),np.zeros((Imax-Imin+1, Jmax-Jmin+1))
        for line in range((Imax-Imin+1)*(Jmax-Jmin+1)*Num_T):
            i,j,k,t = (float(t) for t in re.split(',| |\[|\]|\(|\)',
                                                  fp.readline()[:-1])
                       if len(t)>0)
            Data0p7[int(i),int(j),int(k)] = t
            #TODO一个问题是这个不是中心点，第二个是似乎x和y搞反了
            X_mesh[int(i),int(j)],Y_mesh[int(i),int(j)] = (Imin+i+1/2)/(Imax-Imin+1), (Jmin+j+1/2)/(Jmax-Jmin+1)
        # x=np.linspace((xh - xl)/(x2 - x1 +1)*(0.5),1-(xh - xl)/(x2 - x1 +1)*(0.5),nx)
        # y=np.linspace((yh - yl)/(y2 - y1 +1)*(0.5),1-(yh - yl)/(y2 - y1 +1)*(0.5),ny)
        # X_mesh,Y_mesh = np.meshgrid(x,y)
        # 光子电子离子  ->  电子离子光子
        Data0p7 = Data0p7[:,:,(1,2,0)].reshape(-1,3)
    
    #0.7误差大的原因是这个时刻写的0.3
    X_2d3t_0p7 = np.concatenate([X_mesh.reshape(-1,1),Y_mesh.reshape(-1,1),(0.7)*np.ones(((Imax-Imin+1)*(Jmax-Jmin+1),1))],axis=1)
    upred_0p7 = model.predict(X_2d3t_0p7).reshape(-1,3)
    #np.savetxt('pinnxp%d.txt' % (model.p),X_2d3t)
    upred_0p7 = upred_0p7.detach().numpy()
    
    
    #计算5个时刻的三种范数意义下的误差
    with open('./sol1_wei_aei700_wer_krartr_80_1.txt','r') as fp:
    # with open('./sol1e-8.txt','r') as fp:
        import numpy as np
        from itertools import product
        import re
        DIM, Num_T, _ = eval(fp.readline())
        print(DIM, Num_T, _)
        (Imin,Jmin),(Imax,Jmax),_ = eval(fp.readline())
        print((Imin,Jmin),(Imax,Jmax),_)
        Data1 = np.zeros((Imax-Imin+1, Jmax-Jmin+1, Num_T))
        X_mesh,Y_mesh = np.zeros((Imax-Imin+1, Jmax-Jmin+1)),np.zeros((Imax-Imin+1, Jmax-Jmin+1))
        for line in range((Imax-Imin+1)*(Jmax-Jmin+1)*Num_T):
            i,j,k,t = (float(t) for t in re.split(',| |\[|\]|\(|\)',
                                                  fp.readline()[:-1])
                       if len(t)>0)
            Data1[int(i),int(j),int(k)] = t
            #TODO一个问题是这个不是中心点，第二个是似乎x和y搞反了
            X_mesh[int(i),int(j)],Y_mesh[int(i),int(j)] = (Imin+i+1/2)/(Imax-Imin+1), (Jmin+j+1/2)/(Jmax-Jmin+1)
        # x=np.linspace((xh - xl)/(x2 - x1 +1)*(0.5),1-(xh - xl)/(x2 - x1 +1)*(0.5),nx)
        # y=np.linspace((yh - yl)/(y2 - y1 +1)*(0.5),1-(yh - yl)/(y2 - y1 +1)*(0.5),ny)
        # X_mesh,Y_mesh = np.meshgrid(x,y)
        # 光子电子离子  ->  电子离子光子
        Data1 = Data1[:,:,(1,2,0)].reshape(-1,3)
    
    X_2d3t_1 = np.concatenate([X_mesh.reshape(-1,1),Y_mesh.reshape(-1,1),(1.0)*np.ones(((Imax-Imin+1)*(Jmax-Jmin+1),1))],axis=1)
    upred_1 = model.predict(X_2d3t_1).reshape(-1,3)
    #np.savetxt('pinnxp%d.txt' % (model.p),X_2d3t)
    upred_1 = upred_1.detach().numpy()
    
    
    error_1efu5 = np.abs(Data1efu5-upred_1efu5)
    error_0p3 = np.abs(Data0p3-upred_0p3)
    error_0p5 = np.abs(Data0p5-upred_0p5)
    error_0p7 = np.abs(Data0p7-upred_0p7)
    error_1 = np.abs(Data1-upred_1)
    
    
    #计算每个时刻的误差
    Tel2_err_1efu5,Til2_err_1efu5,Trl2_err_1efu5 = (
        np.sqrt(np.mean(np.square(error_1efu5[:,i])))/np.sqrt(np.mean(np.square(Data1efu5[:,i])))
        for i in range(3))
    Tel1_err_1efu5,Til1_err_1efu5,Trl1_err_1efu5 = (np.mean(error_1efu5[:,i]) for i in range(3))
    Teli_err_1efu5,Tili_err_1efu5,Trli_err_1efu5 = (np.max(error_1efu5[:,i]) for i in range(3))
    
    Tel2_err_0p3,Til2_err_0p3,Trl2_err_0p3 = (
        np.sqrt(np.mean(np.square(error_0p3[:,i])))/np.sqrt(np.mean(np.square(Data0p3[:,i])))
        for i in range(3))
    Tel1_err_0p3,Til1_err_0p3,Trl1_err_0p3 = (np.mean(error_0p3[:,i]) for i in range(3))
    Teli_err_0p3,Tili_err_0p3,Trli_err_0p3 = (np.max(error_0p3[:,i]) for i in range(3))
    
    Tel2_err_0p5,Til2_err_0p5,Trl2_err_0p5 = (
        np.sqrt(np.mean(np.square(error_0p5[:,i])))/np.sqrt(np.mean(np.square(Data0p5[:,i])))
        for i in range(3))
    Tel1_err_0p5,Til1_err_0p5,Trl1_err_0p5 = (np.mean(error_0p5[:,i]) for i in range(3))
    Teli_err_0p5,Tili_err_0p5,Trli_err_0p5 = (np.max(error_0p5[:,i]) for i in range(3))
    
    Tel2_err_0p7,Til2_err_0p7,Trl2_err_0p7 = (
        np.sqrt(np.mean(np.square(error_0p7[:,i])))/np.sqrt(np.mean(np.square(Data0p7[:,i])))
        for i in range(3))
    Tel1_err_0p7,Til1_err_0p7,Trl1_err_0p7 = (np.mean(error_0p7[:,i]) for i in range(3))
    Teli_err_0p7,Tili_err_0p7,Trli_err_0p7 = (np.max(error_0p7[:,i]) for i in range(3))
    
    Tel2_err_1,Til2_err_1,Trl2_err_1 = (
        np.sqrt(np.mean(np.square(error_1[:,i])))/np.sqrt(np.mean(np.square(Data1[:,i])))
        for i in range(3))
    Tel1_err_1,Til1_err_1,Trl1_err_1 = (np.mean(error_1[:,i]) for i in range(3))
    Teli_err_1,Tili_err_1,Trli_err_1 = (np.max(error_1[:,i]) for i in range(3))
    
    print('Tel2_1efu5: %.3e, Til2_1efu5: %.3e, Trl2_1efu5: %.3e' % (Tel2_err_1efu5,Til2_err_1efu5,Trl2_err_1efu5))
    print('Tel1_1efu5: %.3e, Til1_1efu5: %.3e, Trl1_1efu5: %.3e' % (Tel1_err_1efu5,Til1_err_1efu5,Trl1_err_1efu5))
    print('Teli_1efu5: %.3e, Tili_1efu5: %.3e, Trli_1efu5: %.3e' % (Teli_err_1efu5,Tili_err_1efu5,Trli_err_1efu5))
    
    print('\n')
    
    print('Tel2_0p3: %.3e, Til2_0p3: %.3e, Trl2_0p3: %.3e' % (Tel2_err_0p3,Til2_err_0p3,Trl2_err_0p3))
    print('Tel1_0p3: %.3e, Til1_0p3: %.3e, Trl1_0p3: %.3e' % (Tel1_err_0p3,Til1_err_0p3,Trl1_err_0p3))
    print('Teli_0p3: %.3e, Tili_0p3: %.3e, Trli_0p3: %.3e' % (Teli_err_0p3,Tili_err_0p3,Trli_err_0p3))
    
    print('\n')
    
    print('Tel2_0p5: %.3e, Til2_0p5: %.3e, Trl2_0p5: %.3e' % (Tel2_err_0p5,Til2_err_0p5,Trl2_err_0p5))
    print('Tel1_0p5: %.3e, Til1_0p5: %.3e, Trl1_0p5: %.3e' % (Tel1_err_0p5,Til1_err_0p5,Trl1_err_0p5))
    print('Teli_0p5: %.3e, Tili_0p5: %.3e, Trli_0p5: %.3e' % (Teli_err_0p5,Tili_err_0p5,Trli_err_0p5))
    
    print('\n')
    
    print('Tel2_0p7: %.3e, Til2_0p7: %.3e, Trl2_0p7: %.3e' % (Tel2_err_0p7,Til2_err_0p7,Trl2_err_0p7))
    print('Tel1_0p7: %.3e, Til1_0p7: %.3e, Trl1_0p7: %.3e' % (Tel1_err_0p7,Til1_err_0p7,Trl1_err_0p7))
    print('Teli_0p7: %.3e, Tili_0p7: %.3e, Trli_0p7: %.3e' % (Teli_err_0p7,Tili_err_0p7,Trli_err_0p7))
    
    print('\n')
    
    print('Tel2_1: %.3e, Til2_1: %.3e, Trl2_1: %.3e' % (Tel2_err_1,Til2_err_1,Trl2_err_1))
    print('Tel1_1: %.3e, Til1_1: %.3e, Trl1_1: %.3e' % (Tel1_err_1,Til1_err_1,Trl1_err_1))
    print('Teli_1: %.3e, Tili_1: %.3e, Trli_1: %.3e' % (Teli_err_1,Tili_err_1,Trli_err_1))
    
    print('\n')
    
    #计算5个时刻总的误差
    # -1 * 5 * 3
    ustar = np.stack((Data1efu5,Data0p3,Data0p5,Data0p7,Data1),1)
    upred = np.stack((upred_1efu5,upred_0p3,upred_0p5,upred_0p7,upred_1),1)
    error = np.abs(ustar-upred)
    
    Tel2_err,Til2_err,Trl2_err = (
        np.sqrt(np.mean(np.square(error[:,:,i])))/np.sqrt(np.mean(np.square(ustar[:,:,i])))
        for i in range(3))
    Tel1_err,Til1_err,Trl1_err = (np.mean(error[:,:,i]) for i in range(3))
    Teli_err,Tili_err,Trli_err = (np.max(error[:,:,i]) for i in range(3))
    
    print('Tel2: %.3e, Til2: %.3e, Trl2: %.3e' % (Tel2_err,Til2_err,Trl2_err))
    print('Tel1: %.3e, Til1: %.3e, Trl1: %.3e' % (Tel1_err,Til1_err,Trl1_err))
    print('Teli: %.3e, Tili: %.3e, Trli: %.3e' % (Teli_err,Tili_err,Trli_err))
    
    
    # Tel2_err = np.linalg.norm(u_star[:,0:1]-Tep_pred,2)/np.linalg.norm(u_star[:,0:1],2) 
    # Til2_err = np.linalg.norm(u_star[:,1:2]-Tip_pred,2)/np.linalg.norm(u_star[:,1:2],2)
    # Trl2_err = np.linalg.norm(u_star[:,2:3]-Trp_pred,2)/np.linalg.norm(u_star[:,2:3],2)
    

    # Tel1_err = np.linalg.norm(u_star[:,0:1]-Tep_pred,1)/625
    # Til1_err = np.linalg.norm(u_star[:,1:2]-Tip_pred,1)/625
    # Trl1_err = np.linalg.norm(u_star[:,2:3]-Trp_pred,1)/625
    
    # Telinf_err = np.linalg.norm(u_star[:,0:1]-Tep_pred,np.inf)
    # Tilinf_err = np.linalg.norm(u_star[:,1:2]-Tip_pred,np.inf)
    # Trlinf_err = np.linalg.norm(u_star[:,2:3]-Trp_pred,np.inf)    

    
    #画1时刻误差图像
    #我把数据改成这个了，没用之前那个数据，但是应该不影响结果
    with open('./sol1_wei_aei700_wer_krartr_80_1.txt','r') as fp:
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


    #提供初值
    #t=1e-4+1e-8
    X_2d3t_4_8 = np.concatenate([X_mesh.reshape(-1,1),Y_mesh.reshape(-1,1),(1e-4+1e-8)*np.ones(((Imax-Imin+1)*(Jmax-Jmin+1),1))],axis=1)
    upred_4_8 = model.predict(X_2d3t_4_8).reshape(-1,1)
    #np.savetxt('pinnxp%d.txt' % (model.p),X_2d3t)
    upred_4_8np = upred_4_8.detach().numpy()
    np.savetxt('T_1e-4-8.txt',upred_4_8np)


    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(16,12),dpi=100)
    # for ii in range(Data.shape[2]):
    #     plt.subplot(3,3,ii+1)
    #     plt.imshow(Data[:,:,ii].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
    #     plt.colorbar()
    #     plt.subplot(3,3,ii+1+3)
    #     plt.imshow(T_mesh[:,:,ii].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
    #     plt.colorbar()
    #     plt.subplot(3,3,ii+1+6)
    #     plt.imshow((np.abs(Data[:,:,ii]-T_mesh[:,:,ii])).T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
    #     plt.colorbar()
    
    def format_fn(value, tick_number): 
        return f"{value:.4f}"
   
        #t0.9时刻预测解真解误差
    X_2d3t_t005 = np.concatenate([X_mesh.reshape(-1,1),Y_mesh.reshape(-1,1),1.0*np.ones(((Imax-Imin+1)*(Jmax-Jmin+1),1))],axis=1)
    start_time = time.time()                
    upred_t005 = model.predict(X_2d3t_t005).reshape(-1,1)
    elapsed = time.time() - start_time
    print('Inference time: %.4f' % (elapsed))
    upred_t005np = upred_t005.detach().numpy()
    T_mesht005 = upred_t005np.reshape(Data.shape)

    #画参考解和误差
    plt.rcParams['font.size']=12
    maxve = np.max(Data[:,:,0])
    minve = np.min(Data[:,:,0])
    plt.figure(figsize=(11,9),dpi=100)
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
    plt.title("Reference solution Te")
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
    plt.title("Reference solution Ti")
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
    plt.title("Reference solution Tr")


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
    plt.title("Predictive Te")
    plt.subplot(3,3,1+1+3)
    plt.imshow(T_mesht005[:,:,1].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower',vmin=minvi,vmax=maxvi)
    # plt.imshow(T_mesht005[:,:,1].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
    # tip=plt.imshow(T_mesht005[:,:,1].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
    # tip.set_clim(minvi,maxvi)
    cbartip=plt.colorbar()
    cbartip.set_ticks(list(np.linspace(minvi, maxvi, 5)))
    plt.xlabel("$x$", fontsize=12)
    plt.ylabel("$y$", fontsize=12)
    plt.title("Predictive Ti")
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
    plt.title("Predictive Tr")

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
    plt.title("Point-wise Error Te")
    plt.subplot(3,3,1+1+6)
    plt.imshow((np.abs(Data[:,:,1]-T_mesht005[:,:,1])).T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
    plt.colorbar()
    plt.xlabel("$x$", fontsize=12)
    plt.ylabel("$y$", fontsize=12)
    plt.title("Point-wise Error Ti")
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
    plt.title("Point-wise Error Tr")
    plt.suptitle('t=1.0')

    try:
        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, \
            wspace=None, hspace=None)
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, \
        #     wspace=None, hspace=None)

        # plt.subplots(3,3,constrained_layout=True)#改成True后可以自动调整绘图区域在整个figure上的位置
        # plt.tight_layout(pad=1,h_pad=5.0,w_pad=5.0)
        plt.savefig('./figures/'+FolderName+'3x3.pdf',bbox_inches='tight')
    except:
        print('save img failed')
        plt.show()
    plt.close()  # 这是关闭绘图区，自己查看我打算画在一张图上了



    """
    #t0.9时刻预测解真解误差
    X_2d3t_t005 = np.concatenate([X_mesh.reshape(-1,1),Y_mesh.reshape(-1,1),1.0*np.ones(((Imax-Imin+1)*(Jmax-Jmin+1),1))],axis=1)
    upred_t005 = model.predict(X_2d3t_t005).reshape(-1,1)
    upred_t005np = upred_t005.detach().numpy()
    T_mesht005 = upred_t005np.reshape(Data.shape)
    
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
        plt.savefig('./figures/'+FolderName+'3x3.pdf',bbox_inches='tight')
    except:
        print('save img failed')
        plt.show()
    plt.close()  # 这是关闭绘图区，自己查看我打算画在一张图上了
    """ 
    
    #t0时刻预测解
    X_2d3t_t0 = np.concatenate([X_mesh.reshape(-1,1),Y_mesh.reshape(-1,1),0*np.ones(((Imax-Imin+1)*(Jmax-Jmin+1),1))],axis=1)
    upred_t0 = model.predict(X_2d3t_t0).reshape(-1,1)
    upred_t0np = upred_t0.detach().numpy()
    T_mesht0 = upred_t0np.reshape(Data.shape)
    
    plt.figure(figsize=(10,8),dpi=100)
    # plt.figure()
    plt.subplot(1,3,1)
    im1=plt.imshow(T_mesht0[:,:,0].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
    cbar=plt.colorbar(im1,fraction=0.046,pad=0.04)
    # cbar=plt.colorbar()
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    cbar.ax.yaxis.set_major_formatter(ScalarFormatter()) 
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
    plt.xlabel("$x$", fontsize=12)
    plt.ylabel("$y$", fontsize=12)
    plt.title("PINN Te")
    plt.subplot(1,3,2)
    im2=plt.imshow(T_mesht0[:,:,1].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
    cbar=plt.colorbar(im2,fraction=0.046,pad=0.04)
    # cbar=plt.colorbar()
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    cbar.ax.yaxis.set_major_formatter(ScalarFormatter()) 
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
    plt.xlabel("$x$", fontsize=12)
    plt.ylabel("$y$", fontsize=12)
    plt.title("PINN Ti")
    plt.subplot(1,3,3)
    im3=plt.imshow(T_mesht0[:,:,2].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
    cbar=plt.colorbar(im3,fraction=0.046,pad=0.04)
    # cbar=plt.colorbar()
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    cbar.ax.yaxis.set_major_formatter(ScalarFormatter()) 
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
    plt.xlabel("$x$", fontsize=12)
    plt.ylabel("$y$", fontsize=12)
    plt.title("PINN Tr")
    
    try:
        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, \
            wspace=None, hspace=None)
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, \
        #     wspace=None, hspace=None)
        
        # plt.subplots(3,3,constrained_layout=True)#改成True后可以自动调整绘图区域在整个figure上的位置
        # plt.tight_layout(pad=1,h_pad=5.0,w_pad=5.0) 
        plt.savefig('./figures/'+FolderName+'t0_1x3.pdf',bbox_inches='tight')
    except:
        print('save img failed')
        plt.show()
    plt.close()  # 这是关闭绘图区，自己查看我打算画在一张图上了
    
    #t0.1时刻预测解
    X_2d3t_t0dot1 = np.concatenate([X_mesh.reshape(-1,1),Y_mesh.reshape(-1,1),0.7*np.ones(((Imax-Imin+1)*(Jmax-Jmin+1),1))],axis=1)
    upred_t0dot1 = model.predict(X_2d3t_t0dot1).reshape(-1,1)
    upred_t0dot1np = upred_t0dot1.detach().numpy()
    T_mesh0dot1 = upred_t0dot1np.reshape(Data.shape)
    
    plt.figure(figsize=(10,8),dpi=100)
    plt.subplot(1,3,1)
    im1=plt.imshow(T_mesh0dot1[:,:,0].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
    cbar=plt.colorbar(im1,fraction=0.046,pad=0.04)
    # cbar=plt.colorbar()
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    cbar.ax.yaxis.set_major_formatter(ScalarFormatter()) 
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
    plt.xlabel("$x$", fontsize=12)
    plt.ylabel("$y$", fontsize=12)
    plt.title("PINN Te")
    plt.subplot(1,3,2)
    im2=plt.imshow(T_mesh0dot1[:,:,1].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
    cbar=plt.colorbar(im2,fraction=0.046,pad=0.04)
    # cbar=plt.colorbar()
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    cbar.ax.yaxis.set_major_formatter(ScalarFormatter()) 
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
    plt.xlabel("$x$", fontsize=12)
    plt.ylabel("$y$", fontsize=12)
    plt.title("PINN Ti")
    plt.subplot(1,3,3)
    im3=plt.imshow(T_mesh0dot1[:,:,2].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
    cbar=plt.colorbar(im3,fraction=0.046,pad=0.04)
    # cbar=plt.colorbar()
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    cbar.ax.yaxis.set_major_formatter(ScalarFormatter()) 
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
    plt.xlabel("$x$", fontsize=12)
    plt.ylabel("$y$", fontsize=12)
    plt.title("PINN Tr")
    
    try:
        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, \
            wspace=None, hspace=None)
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, \
        #     wspace=None, hspace=None)
        
        # plt.subplots(3,3,constrained_layout=True)#改成True后可以自动调整绘图区域在整个figure上的位置
        # plt.tight_layout(pad=1,h_pad=5.0,w_pad=5.0) 
        plt.savefig('./figures/'+FolderName+'t1_1x3.pdf',bbox_inches='tight')
    except:
        print('save img failed')
        plt.show()
    plt.close()  # 这是关闭绘图区，自己查看我打算画在一张图上了


    # X_2d3t_t1 = np.concatenate([X_mesh.reshape(-1,1),Y_mesh.reshape(-1,1),1*np.ones(((Imax-Imin+1)*(Jmax-Jmin+1),1))],axis=1)
    # upred_t1 = model.predict(X_2d3t_t1).reshape(-1,1)
    # upred_t1 = upred_t1.detach().numpy()
    # T_mesh1 = upred_t1.reshape(Data.shape)
    
    # plt.figure(figsize=(10,8),dpi=100)
    # plt.subplot(1,3,1)
    # im1=plt.imshow(T_mesh1[:,:,0].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
    # cbar=plt.colorbar(im1,fraction=0.046,pad=0.04)
    # # cbar=plt.colorbar()
    # cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # cbar.ax.yaxis.set_major_formatter(ScalarFormatter()) 
    # cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
    # plt.xlabel("$x$", fontsize=12)
    # plt.ylabel("$y$", fontsize=12)
    # plt.title("PINN Te")
    # plt.subplot(1,3,2)
    # im2=plt.imshow(T_mesh1[:,:,1].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
    # cbar=plt.colorbar(im2,fraction=0.046,pad=0.04)
    # # cbar=plt.colorbar()
    # cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # cbar.ax.yaxis.set_major_formatter(ScalarFormatter()) 
    # cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
    # plt.xlabel("$x$", fontsize=12)
    # plt.ylabel("$y$", fontsize=12)
    # plt.title("PINN Ti")
    # plt.subplot(1,3,3)
    # im3=plt.imshow(T_mesh1[:,:,2].T,cmap='jet',extent=(0,1,0,1),aspect='equal',origin='lower')
    # cbar=plt.colorbar(im3,fraction=0.046,pad=0.04)
    # # cbar=plt.colorbar()
    # cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # cbar.ax.yaxis.set_major_formatter(ScalarFormatter()) 
    # cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
    # plt.xlabel("$x$", fontsize=12)
    # plt.ylabel("$y$", fontsize=12)
    # plt.title("PINN Tr")
    
    # try:
    #     plt.tight_layout()
    #     plt.subplots_adjust(left=None, bottom=None, right=None, top=None, \
    #         wspace=None, hspace=None)
    #     # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, \
    #     #     wspace=None, hspace=None)
        
    #     # plt.subplots(3,3,constrained_layout=True)#改成True后可以自动调整绘图区域在整个figure上的位置
    #     # plt.tight_layout(pad=1,h_pad=5.0,w_pad=5.0) 
    #     plt.savefig('./figures/'+FolderName+'t1_1x3.png',bbox_inches='tight')
    # except:
    #     print('save img failed')
    #     plt.show()
    # plt.close()  # 这是关闭绘图区，自己查看我打算画在一张图上了

    # coordx = []
    # coordy = []
    # for i in range(1,nx+1):
    #     valuex = (xh - xl)/(x2 - x1 +1)*(i-0.5)
    #     coordx.append(valuex)
    # for j in range(1,ny+1):
    #     valuey = (yh - yl)/(y2 - y1 +1)*(j-0.5)
    #     coordy.append(valuey)
    # #print(coordx)
    # #X_mesh = [tt.reshape(nx,ny) for tt in np.meshgrid(coordx,coordy,indexing='ij')]
    # X_mesh,Y_mesh = np.meshgrid(coordx,coordy)
    # #X_mesh,Y_mesh = X_mesh,Y_mesh
    # X_2d3t = np.concatenate([X_mesh.reshape(-1,1),Y_mesh.reshape(-1,1)],axis=1)
    # #x0 = np.linspace(0,1,66).reshape(-1,1)
    # upred0 = model.predict(X_2d3t).reshape(-1,1)
    # np.savetxt('pinnxp%d.txt' % (model.p),X_2d3t)
    # upred0np = upred0.detach().numpy()
    # np.savetxt('pinnup%d.txt' % (model.p),upred0np)
    # M,N = 401,401
    
    #给PINN初值,存的还有问题
    # x=np.linspace((xh - xl)/(x2 - x1 +1)*(0.5),1-(xh - xl)/(x2 - x1 +1)*(0.5),nx)
    # y=np.linspace((yh - yl)/(y2 - y1 +1)*(0.5),1-(yh - yl)/(y2 - y1 +1)*(0.5),ny)
    # t=np.array([0.1])
    # X_mesh = [tt.reshape(nx,ny) for tt in np.meshgrid(x,y,t,indexing='ij')]
    # X_plot = np.concatenate([t.reshape(-1,1) for t in X_mesh],axis=1)
    # # Te_plot=Te_solution(*X_mesh)
    # # Ti_plot=Ti_solution(*X_mesh)
    # # Tr_plot=Tr_solution(*X_mesh)
    # u_pred=model.predict(X_plot)
    # upred0np = u_pred.detach().numpy()
    # T_pred = torch.split(model.predict(X_plot),1,1)
    # Tep_pred,Tip_pred,Trp_pred = (tt.reshape(nx,ny).data.numpy() for tt in T_pred)
    # # print(Tep_pred)    
    # np.savetxt('pinnxp31.txt', X_plot)
    # np.savetxt('pinnup31.txt', upred0np)
    

    try:
        os.mkdir('./figures/')
        print('Create Folder: ./figures/')
    except:
        pass
    try:
        os.mkdir('./figures/'+FolderName)
        print('Create Folder: ./figures/'+FolderName)
    except:
        pass
    #plot
    plt.figure('PDE Loss Curve',figsize=(12,5),dpi=150)
    #print(rank,full_MSE_hist)
    plt.plot(200*np.arange(0,len(Teloss)), Teloss,  'r.-', linewidth = 1,  label = 'Te') 
    plt.plot(200*np.arange(0,len(Tiloss)), Tiloss,  'g--', linewidth = 1,  label = 'Ti')  
    plt.plot(200*np.arange(0,len(Trloss)), Trloss,  'b-', linewidth = 1,  label = 'Tr')        
    plt.xlabel('$\#$ iterations')
    plt.ylabel('Loss')
    plt.title('PINN 2D3T PDE Loss', fontsize = 12)
    plt.yscale('log')
    plt.legend(loc='upper right') 
    try:
        plt.savefig('./figures/'+FolderName+'Loss.pdf',bbox_inches='tight')
    except:
        print('save img failed')
        plt.show()
    plt.close()  # 这是关闭绘图区，自己查看我打算画在一张图上了
    
    #初值条件损失
    plt.figure('Init and bound Loss Curve',figsize=(12,5),dpi=150)
    #print(rank,full_MSE_hist)
    plt.plot(200*np.arange(0,len(Teiniloss)), Teiniloss,  'r.-', linewidth = 1,  label = 'Teinit') 
    plt.plot(200*np.arange(0,len(Tiiniloss)), Tiiniloss,  'g--', linewidth = 1,  label = 'Tiinit')  
    plt.plot(200*np.arange(0,len(Triniloss)), Triniloss,  'b-', linewidth = 1,  label = 'Trinit')   
    plt.plot(200*np.arange(0,len(Trboundloss)), Trboundloss,  'fuchsia', linewidth = 1,  label = 'Trbound')  
    plt.xlabel('$\#$ iterations')
    plt.ylabel('Init Loss')
    plt.title('PINN 2D3T Init and bound Loss', fontsize = 12)
    plt.yscale('log')
    plt.legend(loc='upper right') 
    try:
        plt.savefig('./figures/'+FolderName+'Init_bound_Loss.pdf',bbox_inches='tight')
    except:
        print('save img failed')
        plt.show()
    plt.close()  # 这是关闭绘图区，自己查看我打算画在一张图上了
    
    
    #L2相对误差曲线
    #这里L2误差用的是之前精确解给的u_star，画错了
    # plt.figure('Rel E2 Curve',figsize=(12,5),dpi=150)
    # #print(rank,full_MSE_hist)
    # #fig, ax = newfig(1.0, 1.1)
    # plt.plot(20*np.arange(0,len(TeL2)), TeL2,  'r.-', linewidth = 1, label = 'Te') 
    # plt.plot(20*np.arange(0,len(TeL2)), TiL2,  'g--', linewidth = 1, label = 'Ti') 
    # plt.plot(20*np.arange(0,len(TeL2)), TrL2,  'b-', linewidth = 1, label = 'Tr') 
    # plt.xlabel('$\#$ iterations')
    # plt.ylabel('Rel. $L_2$ Error')
    # plt.title('PINN 2D3T Rel. $L_2$ Error', fontsize = 12)
    # plt.yscale('log')
    # plt.legend(loc='upper right')
    # try:
    #     plt.savefig('./figures/'+FolderName+'L2_error.png',bbox_inches='tight')
    # except:
    #     print('save img failed')
    #     plt.show()
    # plt.close()  # 这是关闭绘图区，自己查看我打算画在一张图上了
    
    
    
    # #plot solution
    # #*************** Exact solution
    # M,N = 401,401
    # x=np.linspace(0,1,M)
    # y=np.linspace(0,1,N)
    # t=np.array([0.1])
    # X_mesh = [tt.reshape(M,N) for tt in np.meshgrid(x,y,t,indexing='ij')]
    # X_plot = np.concatenate([t.reshape(-1,1) for t in X_mesh],axis=1)
    # Te_plot=Te_solution(*X_mesh)
    # Ti_plot=Ti_solution(*X_mesh)
    # Tr_plot=Tr_solution(*X_mesh)
    # u_pred=model.predict(X_plot)
    # T_pred = torch.split(model.predict(X_plot),1,1)
    # Tep_pred,Tip_pred,Trp_pred = (tt.reshape(M,N).data.numpy() for tt in T_pred)
    # # Tep_pred = Tep_pred.data.numpy().reshape(-1,1)
    # # Tip_pred = Tip_pred.data.numpy().reshape(-1,1)
    # # Trp_pred = Trp_pred.data.numpy().reshape(-1,1)

    # img_kwargs = {'interpolation':'hermite', 
    #               'cmap':'jet',
    #               'extent':[lb[1],ub[1],lb[0],ub[0]],
    #               'origin':'lower',
    #               'aspect':'equal'}

    # plt.figure('PINNs 2D3T Solution',figsize=(12,3.2),dpi=150)
    # plt.suptitle('PINNs 2D3T Solution', fontsize=12)
    # plt.subplot(1,3,1)
    # plt.imshow(Tep_pred, **img_kwargs)
    # plt.colorbar()
    # plt.xlabel('$x$', fontsize=12)
    # plt.ylabel('$y$', fontsize=12)
    # plt.title('Te')
    # plt.subplot(1,3,2)
    # plt.imshow(Tip_pred, **img_kwargs)
    # plt.colorbar()
    # plt.xlabel('$x$', fontsize=12)
    # plt.ylabel('$y$', fontsize=12)
    # plt.title('Ti')
    # plt.subplot(1,3,3)
    # plt.imshow(Trp_pred, **img_kwargs)
    # plt.colorbar()
    # plt.xlabel('$x$', fontsize=12)
    # plt.ylabel('$y$', fontsize=12)
    # plt.title('Tr')
    # plt.tight_layout()
    # try:
    #     plt.savefig('./figures/'+FolderName+'pinns_solution.png',bbox_inches='tight')
    # except:
    #     print('save img failed')
    #     plt.show()
    # plt.close()  # 这是关闭绘图区，自己查看我打算画在一张图上了

    # plt.figure('Truth 2D3T Solution',figsize=(12,3.2),dpi=150)
    # plt.suptitle('Truth 2D3T Solution', fontsize=12)
    # plt.subplot(1,3,1)
    # plt.imshow(Te_plot, **img_kwargs)
    # plt.colorbar()
    # plt.xlabel('$x$', fontsize=12)
    # plt.ylabel('$y$', fontsize=12)
    # plt.title('Te')
    # plt.subplot(1,3,2)
    # plt.imshow(Ti_plot, **img_kwargs)
    # plt.colorbar()
    # plt.xlabel('$x$', fontsize=12)
    # plt.ylabel('$y$', fontsize=12)
    # plt.title('Ti')
    # plt.subplot(1,3,3)
    # plt.imshow(Tr_plot, **img_kwargs)
    # plt.colorbar()
    # plt.xlabel('$x$', fontsize=12)
    # plt.ylabel('$y$', fontsize=12)
    # plt.title('Tr')
    # plt.tight_layout()
    # try:
    #     plt.savefig('./figures/'+FolderName+'truth_solution.png',bbox_inches='tight')
    # except:
    #     print('save img failed')
    #     plt.show()
    # plt.close()  # 这是关闭绘图区，自己查看我打算画在一张图上了


    # plt.figure('Point-wise Error',figsize=(12,3.2),dpi=150)
    # plt.suptitle('PINN 2D3T Point-wise error', fontsize=12)
    # plt.subplot(1,3,1)
    # plt.imshow(np.abs(Te_plot-Tep_pred), **img_kwargs)
    # plt.colorbar()
    # plt.xlabel('$x$', fontsize=12)
    # plt.ylabel('$y$', fontsize=12)
    # plt.title('Te')
    # plt.subplot(1,3,2)
    # plt.imshow(np.abs(Ti_plot-Tip_pred), **img_kwargs)
    # plt.colorbar()
    # plt.xlabel('$x$', fontsize=12)
    # plt.ylabel('$y$', fontsize=12)
    # plt.title('Ti')
    # plt.subplot(1,3,3)
    # plt.imshow(np.abs(Tr_plot-Trp_pred), **img_kwargs)
    # plt.colorbar()
    # plt.xlabel('$x$', fontsize=12)
    # plt.ylabel('$y$', fontsize=12)
    # plt.title('Tr')
    # plt.tight_layout()
    # try:
    #     plt.savefig('./figures/'+FolderName+'pointwise_error.png',bbox_inches='tight')
    # except:
    #     print('save img failed')
    #     plt.show()
    # plt.close()  # 这是关闭绘图区，自己查看我打算画在一张图上了

    # plt.figure('Summary',figsize=(12,6),dpi=150)
    # plt.subplot(3,5,3)
    # plt.imshow(Tep_pred, **img_kwargs)
    # plt.colorbar()
    # plt.xlabel('$x$', fontsize=12)
    # plt.ylabel('$y$', fontsize=12)
    # plt.title('PINNs Te')
    # plt.subplot(3,5,4)
    # plt.imshow(Tip_pred, **img_kwargs)
    # plt.colorbar()
    # plt.xlabel('$x$', fontsize=12)
    # plt.ylabel('$y$', fontsize=12)
    # plt.title('PINNs Ti')
    # plt.subplot(3,5,5)
    # plt.imshow(Trp_pred, **img_kwargs)
    # plt.colorbar()
    # plt.xlabel('$x$', fontsize=12)
    # plt.ylabel('$y$', fontsize=12)
    # plt.title('PINNs Tr')
    # plt.subplot(3,5,8)
    # plt.imshow(Te_plot, **img_kwargs)
    # plt.colorbar()
    # plt.xlabel('$x$', fontsize=12)
    # plt.ylabel('$y$', fontsize=12)
    # plt.title('True solution Te')
    # plt.subplot(3,5,9)
    # plt.imshow(Ti_plot, **img_kwargs)
    # plt.colorbar()
    # plt.xlabel('$x$', fontsize=12)
    # plt.ylabel('$y$', fontsize=12)
    # plt.title('True solution Ti')
    # plt.subplot(3,5,10)
    # plt.imshow(Tr_plot, **img_kwargs)
    # plt.colorbar()
    # plt.xlabel('$x$', fontsize=12)
    # plt.ylabel('$y$', fontsize=12)
    # plt.title('True solution Tr')
    # plt.subplot(3,5,13)
    # plt.imshow(np.abs(Te_plot-Tep_pred), **img_kwargs)
    # plt.colorbar()
    # plt.xlabel('$x$', fontsize=12)
    # plt.ylabel('$y$', fontsize=12)
    # plt.title('Error Te')
    # plt.subplot(3,5,14)
    # plt.imshow(np.abs(Ti_plot-Tip_pred), **img_kwargs)
    # plt.colorbar()
    # plt.xlabel('$x$', fontsize=12)
    # plt.ylabel('$y$', fontsize=12)
    # plt.title('Error Ti')
    # plt.subplot(3,5,15)
    # plt.imshow(np.abs(Tr_plot-Trp_pred), **img_kwargs)
    # plt.colorbar()
    # plt.xlabel('$x$', fontsize=12)
    # plt.ylabel('$y$', fontsize=12)
    # plt.title('Error Tr')
    # plt.tight_layout()

    # plt.subplot(2,5,(1,2))
    # plt.plot(20*np.arange(0,len(Teloss)), Teloss,  'r.-', linewidth = 1,  label = 'Te') 
    # plt.plot(20*np.arange(0,len(Tiloss)), Tiloss,  'g--', linewidth = 1,  label = 'Ti')  
    # plt.plot(20*np.arange(0,len(Trloss)), Trloss,  'b-', linewidth = 1,  label = 'Tr')           
    # plt.xlabel('$\#$ iterations')
    # plt.ylabel('Loss')
    # plt.title('PINN 2D3T Loss', fontsize = 12)
    # plt.yscale('log')
    # plt.legend(loc='upper right') 
    # plt.subplot(2,5,(6,7))
    # plt.plot(20*np.arange(0,len(TeL2)), TeL2,  'r.-', linewidth = 1, label = 'Te') 
    # plt.plot(20*np.arange(0,len(TeL2)), TiL2,  'g--', linewidth = 1, label = 'Ti') 
    # plt.plot(20*np.arange(0,len(TeL2)), TrL2,  'b-', linewidth = 1, label = 'Tr') 
    # plt.xlabel('$\#$ iterations')
    # plt.ylabel('Rel. $L_2$ Error')
    # plt.title('PINN 2D3T Rel. $L_2$ Error', fontsize = 12)
    # plt.yscale('log')
    # plt.legend(loc='upper right')
    # plt.tight_layout()
    # try:
    #     plt.savefig('./figures/'+FolderName+'summary.png',bbox_inches='tight')
    # except:
    #     print('save img failed')
    #     plt.show()


    """
    # fig, ax = plt.newfig(1.0, 1.1)
    fig = plt.figure()
    gridspec.GridSpec(1,1)

    ax = plt.subplot2grid((1,1), (0,0))
    maxLevel = max(max(u_plote))
    minLevel = min(min(u_plote))
    levels = np.linspace(minLevel-0.01, maxLevel+0.01, 200)
    CS_ext1 = ax.contourf(x, y, u_plote, levels=levels, cmap='jet', origin='lower')

    cbar = fig.colorbar(CS_ext1)
    #                        , ticks=[-1, -0.5, 0, 0.5, 1])
    #    cbar.ax.set_yticklabels(['-1', '-0.5', '0', '0.5', '1'])


    cbar.ax.tick_params(labelsize=20)
    ax.set_xlim(-0.01, 1)
    ax.set_ylim(-1.01, 1.02)

        #ax_pred.locator_params(nbins=5)
        #ax_pred.set_xticklabels(np.linspace(0,1,5), rotation=0, fontsize=18)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    ax.set_title('$ u^{Exact} $')

    #    for xc in x_interface:
    #        ax.axhline(y=xc, linewidth=1, color = 'w')
        
    fig.tight_layout()

    fig.set_size_inches(w=15, h=8)
    plt.savefig('./figures/Exact-solution')    #KdV3SD_PredPlot')
    """
    
    
    #plot predict solution
