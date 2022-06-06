import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from sklearn.decomposition import PCA
import copy

def mul(net1, net2):
    result = pickle.loads(pickle.dumps(net1))
    net1_dict = net1.state_dict()
    net2_dict = net2.state_dict()
    weight_keys = list(net1_dict.keys())
    result_dict = result.state_dict()
    for key in weight_keys:
        result_dict[key] = net1_dict[key] * net2_dict[key]
    result.load_state_dict(result_dict)
    return result

def scalar_mul(scalar_list, net_list, rg=False):
    result = copy.deepcopy(net_list[0])
    with torch.no_grad():
        result_dict = result.state_dict()
        nets_dict = [neti.state_dict() for neti in net_list]
        weight_keys = list(result_dict.keys())
        for key in weight_keys:
            result_dict[key] *= 0
            for j in range(len(scalar_list)):
                result_dict[key] = result_dict[key] + scalar_list[j]*nets_dict[j][key]
        result.load_state_dict(result_dict)
        return result

def scalar_mul_no_buffer(scalar_list, net_list, rg=False):
    result = copy.deepcopy(net_list[0])#pickle.loads(pickle.dumps(net_list[0]))
    with torch.no_grad():
        worker_params = [list(x.parameters()) for x in net_list]
        for i, params in enumerate(result.parameters()):
            params.data = 0 * params.data
            for j in range(len(scalar_list)):
                params.data = params.data + worker_params[j][i] * scalar_list[j]
        return result

def zero_weights(net):
    with torch.no_grad():
        for param in net.parameters():
            param.data *= 0

def inner_p(net1, net2, rg=False):
    result = torch.zeros(1)[0]
    if rg:
        result.requires_grad = True
    if next(net1.parameters()).is_cuda:
        #print('something is on cuda')
        result = result.cuda()
    net1_params = list(net1.parameters())
    net2_params = list(net2.parameters())

    for i in range(len(net1_params)):
        result = result + (net1_params[i]*net2_params[i]).sum()
    #print("inner product: {}".format(result))
    #print(result)
    return result

def lin_over_sqrt_epsilon(net1, net2, beta1, beta2, t, epsilon):
    result = pickle.loads(pickle.dumps(net1))
    net1_dict = net1.state_dict()
    net2_dict = net2.state_dict()
    weight_keys = list(net1_dict.keys())
    result_dict = result.state_dict()
    for key in weight_keys:
        result_dict[key] = (net1_dict[key]/(1-beta1**t))/ (torch.sqrt(net2_dict[key]/(1-beta2**t))+epsilon)
    result.load_state_dict(result_dict)
    return result


def net_flatten(net):
    newnet = pickle.loads(pickle.dumps(net))
    n_dict = newnet.state_dict()
    w_keys = list(n_dict.keys())
    res = torch.flatten(n_dict[w_keys[0]])
    for i in range(1,len(w_keys)):
        res = torch.cat((res,torch.flatten(n_dict[w_keys[0]])))
    return res


def gradient_flatten(net):
    res = torch.tensor([])#torch.flatten(n_dict[w_keys[0]])
    for p in net.parameters():
        res = torch.cat((res,torch.flatten(p.grad.data)))
    return res


def reconstruct_gradient(net):
    res = copy.deepcopy(net)
    with torch.no_grad():
        for (p1,p2) in zip(res.parameters(), net.parameters()):
            p1.data *= 0
            p1.data += p2.grad.data
    return res


def reset(model):
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()

def gaussian_sampling(dwt, model):
    coefs = torch.ones(len(dwt))/len(dwt)
    average_dw = scalar_mul(coefs, dwt)
    centralized_dw = [scalar_mul([1, -1], [dw, average_dw]) for dw in dwt]
    var = torch.tensor([inner_p(cdw, cdw) for cdw in centralized_dw])
    std = torch.sqrt(var.mean())
    deltaw = copy.deepcopy(model)
    reset(deltaw)
    norm = inner_p(deltaw, deltaw)
    print("norm is {}".format(norm))
    z = torch.randn(1).item()
    #z = 0
    return scalar_mul([1,1,std/torch.sqrt(norm)*z], [model,average_dw, deltaw])


def add(net1, net2):
    result = pickle.loads(pickle.dumps(net1))
    net1_dict = net1.state_dict()
    net2_dict = net2.state_dict()
    weight_keys = list(net1_dict.keys())
    result_dict = result.state_dict()
    for key in weight_keys:
        result_dict[key] = net1_dict[key] + net2_dict[key]
    result.load_state_dict(result_dict)
    return result

def ada_update(v_vec, m_vec, model_list, posterior_model_list, eta, count,sld, args):
    #new_model_list = copy.deepcopy(), v_vec, m_vec
    dm = [scalar_mul([1,-1],[posterior_model_list[i],model_list[i]]) for i in range(len(model_list))]
    beta1 = args['server-beta1']
    beta2 = args['server-beta2']
    epsilon = args['server-epsilon']

    v_vec_new = copy.deepcopy(v_vec)

    m_vec_new = [scalar_mul([beta1,1-beta1],[m_vec[i], dm[i]]) if i in sld else m_vec[i] for i in range(len(model_list))]
    if args['fed'] in {'bfed-adam','fed-adam'}:
        v_vec_new = [scalar_mul([beta2,1-beta2], [v_vec[i], mul(dm[i],dm[i])]) if i in sld else v_vec[i] for i in range(len(model_list))]
    elif args['fed'] in {'bfed-yogi', 'fed-yogi'}:
        for j in range(len(model_list)):
            if j in sld:
                dmjpara = list(dm[j].parameters())
                vj = list(v_vec[j].parameters())
                with torch.no_grad():
                    for i, params in enumerate(v_vec_new[j].parameters()):
                        params.data = 0 * params.data
                        dsq = (dmjpara[i].data) ** 2
                        vji = vj[i].data
                        params.data = vji - (1-beta2)*dsq*torch.sign(vji-dsq)
            else:
                v_vec_new[j] = v_vec[j]

    dw = [lin_over_sqrt_epsilon(m_vec_new[i], v_vec_new[i], beta1, beta2, count[i], epsilon) for i in range(len(model_list))]

    new_model_list = [scalar_mul([1,eta],[posterior_model_list[i],dw[i]]) if i in sld else posterior_model_list[i] for i in range(len(model_list))]
    return new_model_list, v_vec_new, m_vec_new

def ada_update_single(v, m, model, dw, eta, t, args):
    beta1 = args['server-beta1']
    beta2 = args['server-beta2']
    epsilon = args['server-epsilon']
    #print(len(dw))
    #print(len(m))
    m_new = scalar_mul([beta1, 1 - beta1], [m, dw])
    v_new = copy.deepcopy(v)
    if args['fed'] in {'bfed-adam', 'fed-adam'}:
        v_new = scalar_mul([beta2, 1 - beta2], [v, mul(dw, dw)])
    elif args['fed'] in {'bfed-yogi', 'fed-yogi'}:
        with torch.no_grad():
            dwlist = list(dw.parameters())
            vlist = list(v.parameters())
            for i, params in enumerate(v_new.parameters()):
                params.data = 0 * params.data
                dsq = (dwlist[i].data) ** 2
                vji = vlist[i].data
                params.data = vji - (1 - beta2) * dsq * torch.sign(vji - dsq)
    dwt = lin_over_sqrt_epsilon(m_new, v_new, beta1, beta2, t, epsilon)
    new_model = scalar_mul([1, eta], [model, dwt])
    return new_model, v_new, m_new


def gradient_l2_norm(model):
    norm = 0
    for p in model.parameters():
        norm += p.grad.data.norm(2).item() ** 2
    return norm

def gradient_norm_lw(model):
    res = []
    for p in model.parameters():
        res.append(p.grad.data.norm(2).item())
    return res

def model_std(model_list):
    new_list = pickle.loads(pickle.dumps(model_list))
    wt = np.zeros(len(new_list))
    center_of_mass = scalar_mul(wt,new_list)
    var = 0
    for i in new_list:
        diff = scalar_mul([1,-1],[i,center_of_mass])
        var = var + inner_p(diff,diff)
    return torch.sqrt(var/(len(model_list)-1))


def model_star(model_list):
    res = []
    model_belong_dict = dict()
    lth = len(model_list)
    max_index = 0
    for i in range(lth):
        for j in range(0,i):
            diff = scalar_mul([1,-1],[model_list[i],model_list[j]])
            dis = torch.sqrt(inner_p(diff, diff))
            if dis<1e-1:
                if j in model_belong_dict.keys():
                    model_belong_dict[i] = copy.deepcopy(model_belong_dict[j])
                else:
                    model_belong_dict[i] = max_index
                    max_index += 1
            res.append(dis.item())
    return res, model_belong_dict

def weight_params_pca(model_list, rank, decentralize=True):
    A = net_flatten(model_list[0]).unsqueeze(0)
    for i in range(1,len(model_list)):
        A = torch.cat((A,net_flatten(model_list[i]).unsqueeze(0)),0)
    A=A.numpy()
    if decentralize:
        mean = A.mean(0)
        A = A - mean
    transformer = PCA(n_components=rank, random_state=0)
    transformer.fit(A)

    #U, S, V = torch.pca_lowrank(A,q=rank)
    return transformer.explained_variance_ratio_

def grad_over_model(numerator, denominator):
    result = pickle.loads(pickle.dumps(denominator))
    #denominator.zero_grad()
    numerator.backward()
    res_params = list(result.parameters())
    deno_params = list(denominator.parameters())
    for i in range(len(res_params)):
        res_params[i] = deno_params[i].grad
    result.zero_grad()
    denominator.zero_grad()
    return result


def rbk(net1, net2, r, rg=False):
    diff = scalar_mul([1,-1],[net1,net2])
    l2norm = inner_p(diff, diff, rg)
    return (-r*l2norm).exp()

def nabla_1rbk(net1, net2, r):
    exp = rbk(net1,net2,r)
    return scalar_mul([-2*exp*r, 2*exp*r], [net1, net2])

if __name__ == '__main__':
    from network import CNNMnist, Twohiddenlayerfc
    ml = []
    for i in range(10):
        ml.append(Twohiddenlayerfc())
    #ms = model_star(ml)
    import copy
    m11=copy.deepcopy(ml[0])
    res = m11(torch.randn(10))
    res = res.sum()
    res.backward()
    ms = weight_params_pca(ml,7)
    print('eigen values: {}'.format(ms))
'''
    n1 = Twohiddenlayerfc()
    n2 = Twohiddenlayerfc()
    print(inner_p(n1,n1))


    n3 = mul(n1,n2)
    input = torch.ones(10)
    print(n3(input))
    print('n3')
    #print(n3.state_dict())
    print(inner_p(n1, n1))
    print(inner_p(n2, n2))
    print(inner_p(n3,n3))
    #print(list(n3.parameters())[1])
    n4 = scalar_mul([10],[n3])
    #print(list(n4.parameters())[1])
    #
    #for p in n3.parameters():
    #    print(p)
    #    p.data = p.data*10
    res = inner_p(n4, n4)
    res.backward()
    #print(inner_p(n4, n4))
    p = n3.parameters()

'''
