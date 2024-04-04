from .tensorBase import *
import tinycudann as tcnn
import torch.nn.functional as F

class TensorVM(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVM, self).__init__(aabb, gridSize, device, **kargs)
        
        

    def init_svd_volume(self, res, device):
        self.plane_coef = torch.nn.Parameter(
            0.1 * torch.randn((3, self.app_n_comp + self.density_n_comp, res, res), device=device))
        self.line_coef = torch.nn.Parameter(
            0.1 * torch.randn((3, self.app_n_comp + self.density_n_comp, res, 1), device=device))
        self.basis_mat = torch.nn.Linear(self.app_n_comp * 3, self.app_dim, bias=False, device=device)

    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.line_coef, 'lr': lr_init_spatialxyz}, 
                     {'params': self.plane_coef, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(), 'lr':lr_init_network},{"params":self.density_encoding[0].parameters(),"lr":lr_init_spatialxyz},{"params":self.density_encoding[1].parameters(),"lr":lr_init_spatialxyz},{"params":self.density_encoding[2].parameters(),"lr":lr_init_spatialxyz}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def compute_features(self, xyz_sampled):

        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach()
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach()

        plane_feats = F.grid_sample(self.plane_coef[:, -self.density_n_comp:], coordinate_plane, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        line_feats = F.grid_sample(self.line_coef[:, -self.density_n_comp:], coordinate_line, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        
        sigma_feature = torch.sum(plane_feats * line_feats, dim=0)
        
        
        plane_feats = F.grid_sample(self.plane_coef[:, :self.app_n_comp], coordinate_plane, align_corners=True).view(3 * self.app_n_comp, -1)
        line_feats = F.grid_sample(self.line_coef[:, :self.app_n_comp], coordinate_line, align_corners=True).view(3 * self.app_n_comp, -1)
        
        
        app_features = self.basis_mat((plane_feats * line_feats).T)
        
        return sigma_feature, app_features

    def compute_densityfeature(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_feats = F.grid_sample(self.plane_coef[:, -self.density_n_comp:], coordinate_plane, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        line_feats = F.grid_sample(self.line_coef[:, -self.density_n_comp:], coordinate_line, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        
        sigma_feature = torch.sum(plane_feats * line_feats, dim=0)
        
        
        return sigma_feature
    
    def compute_appfeature(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        
        plane_feats = F.grid_sample(self.plane_coef[:, :self.app_n_comp], coordinate_plane, align_corners=True).view(3 * self.app_n_comp, -1)
        line_feats = F.grid_sample(self.line_coef[:, :self.app_n_comp], coordinate_line, align_corners=True).view(3 * self.app_n_comp, -1)
        
        
        app_features = self.basis_mat((plane_feats * line_feats).T)
        
        
        return app_features
    

    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            # print(self.line_coef.shape, vector_comps[idx].shape)
            n_comp, n_size = vector_comps[idx].shape[:-1]
            
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            # print(vector_comps[idx].shape, vector_comps[idx].view(n_comp,n_size).transpose(-1,-2).shape, dotp.shape)
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            # print(vector_comps[idx].shape, vector_comps[idx].view(n_comp,n_size).transpose(-1,-2).shape, dotp.shape,non_diagonal.shape)
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        
        return self.vectorDiffs(self.line_coef[:,-self.density_n_comp:]) + self.vectorDiffs(self.line_coef[:,:self.app_n_comp])
    
    
    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        # plane_coef[0] = torch.nn.Parameter(
        #     F.interpolate(plane_coef[0].data, size=(res_target[1], res_target[0]), mode='bilinear',
        #                   align_corners=True))
        # line_coef[0] = torch.nn.Parameter(
        #     F.interpolate(line_coef[0].data, size=(res_target[2], 1), mode='bilinear', align_corners=True))
        # plane_coef[1] = torch.nn.Parameter(
        #     F.interpolate(plane_coef[1].data, size=(res_target[2], res_target[0]), mode='bilinear',
        #                   align_corners=True))
        # line_coef[1] = torch.nn.Parameter(
        #     F.interpolate(line_coef[1].data, size=(res_target[1], 1), mode='bilinear', align_corners=True))
        # plane_coef[2] = torch.nn.Parameter(
        #     F.interpolate(plane_coef[2].data, size=(res_target[2], res_target[1]), mode='bilinear',
        #                   align_corners=True))
        # line_coef[2] = torch.nn.Parameter(
        #     F.interpolate(line_coef[2].data, size=(res_target[0], 1), mode='bilinear', align_corners=True))

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        # self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        # self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

        scale = res_target[0]/self.line_coef.shape[2] #assuming xyz have the same scale
        plane_coef = F.interpolate(self.plane_coef.detach().data, scale_factor=scale, mode='bilinear',align_corners=True)
        line_coef  = F.interpolate(self.line_coef.detach().data, size=(res_target[0],1), mode='bilinear',align_corners=True)
        self.plane_coef, self.line_coef = torch.nn.Parameter(plane_coef), torch.nn.Parameter(line_coef)
        self.compute_stepSize(res_target)
        print(f'upsamping to {res_target}')

from torch import nn
class Encoding(nn.Module):
    def __init__(self,in_channel,config):
        super().__init__()
        self.enc = tcnn.Encoding(in_channel,config)
    def forward(self, x):
        return self.enc(x)


def positional_encoding(positions, freqs):
    
        freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts

class MLPRender(torch.nn.Module):
    def __init__(self,inChanel, output_channel,viewpe=6, featureC=64):
        super(MLPRender, self).__init__()

        self.in_mlpC = inChanel
        self.viewpe = viewpe
        
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,output_channel)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, features):
        sigma = self.mlp(features)
        # sigma = torch.sigmoid(rgb)
        # sigma = F.relu(sigma)

        return sigma

class MLPRender_PE(torch.nn.Module): # position(x,y) -> feature
    def __init__(self, inChanel, output_channel=16, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC =  (2+2*pospe*2) #
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,output_channel)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts):
        indata = [pts]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        mlp_in = torch.cat(indata, dim=-1)
        feature = self.mlp(mlp_in)
        #feature = F.relu(feature) #使用relu激活，因为我们的任务就是预测出体密度的特征，至于这个特征是否需要负数？ 可能需要实验
        

        return feature
    

class MLPRender_PE_Res(torch.nn.Module): # position(x,y) -> positional encodding + ngp_result -> final_feature
    def __init__(self, inChanel, output_channel=16, pospe=6, featureC=128):
        super(MLPRender_PE_Res, self).__init__()

        self.in_mlpC =  (2*pospe*2+inChanel) #(x,y) + positional encoding + ngp_feature
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,output_channel)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, ngp_feature):
        indata = [pts,ngp_feature]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        mlp_in = torch.cat(indata, dim=-1)
        feature = self.mlp(mlp_in)
        #feature = F.relu(feature) #使用relu激活，因为我们的任务就是预测出体密度的特征，至于这个特征是否需要负数？ 可能需要实验
        

        return feature


class TensorVMSplit(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        
        self.ngp_density_encoding_config1 = { #xy
            "n_levels": 1,
            "otype": "HashGrid",
            "n_features_per_level": 16,
            "base_resolution": 300,
            "log2_hashmap_size": 17,
        }
        self.ngp_density_encoding_config2 = { #xz
            "n_levels": 1,
            "otype": "HashGrid",
            "n_features_per_level": 16,
            "base_resolution": 300,
            "log2_hashmap_size": 17,
        }
        self.ngp_density_encoding_config3 = { #yz
            "n_levels": 1,
            "otype": "HashGrid",
            "n_features_per_level": 16,
            "base_resolution": 300,
            "log2_hashmap_size": 17,
        }
        self.ngp_rgb_encoding_config1 = { #xy
            "n_levels": 3,
            "otype": "HashGrid",
            "n_features_per_level": 16,
            "base_resolution": 300,
            "log2_hashmap_size": 17,
        }
        self.ngp_rgb_encoding_config2 = { #xz
            "n_levels": 3,
            "otype": "HashGrid",
            "n_features_per_level": 16,
            "base_resolution": 300,
            "log2_hashmap_size": 17,
        }
        self.ngp_rgb_encoding_config3 = { #yz
            "n_levels": 3,
            "otype": "HashGrid",
            "n_features_per_level": 16,
            "base_resolution": 300,
            "log2_hashmap_size": 17,
        }
        self.newwork_config = {
            "otype": "FullyFusedMLP",
		    "activation": "Relu",
		    "output_activation": "Softplus",
		    "n_neurons": 64,
		    "n_hidden_layers": 2,
        }
        self.ngp_density_encoding = [tcnn.Encoding(2, self.ngp_density_encoding_config1,dtype=torch.float32),tcnn.Encoding(2, self.ngp_density_encoding_config2,dtype=torch.float32),tcnn.Encoding(2, self.ngp_density_encoding_config3,dtype=torch.float32)]
        self.ngp_rgb_encoding = [tcnn.Encoding(2, self.ngp_rgb_encoding_config1,dtype=torch.float32),tcnn.Encoding(2, self.ngp_rgb_encoding_config2,dtype=torch.float32),tcnn.Encoding(2, self.ngp_rgb_encoding_config3,dtype=torch.float32)]
        #print(self.ngp_density_encoding[0].parameters())
        self.feature_num1 = self.ngp_density_encoding_config1["n_levels"] * self.ngp_density_encoding_config1["n_features_per_level"]
        self.feature_num2 = self.ngp_density_encoding_config2["n_levels"] * self.ngp_density_encoding_config2["n_features_per_level"]
        self.feature_num3 = self.ngp_density_encoding_config3["n_levels"] * self.ngp_density_encoding_config3["n_features_per_level"]
        #self.density_network = [MLPRender(self.feature_num1,16).to(device),MLPRender(self.feature_num2,16).to(device),MLPRender(self.feature_num3,16).to(device)]
        self.extra_mlp = [MLPRender_PE_Res(2+self.feature_num1,16).to(device),MLPRender_PE_Res(2+self.feature_num1,16).to(device),MLPRender_PE_Res(2+self.feature_num1,16).to(device)]
        
        #_,self.ngp_line = self.init_one_svd([feature_num,feature_num,feature_num], self.gridSize, 0.1, device)
        self.feature_num4 = self.ngp_rgb_encoding_config1["n_levels"] * self.ngp_rgb_encoding_config1["n_features_per_level"]
        self.feature_num5 = self.ngp_rgb_encoding_config2["n_levels"] * self.ngp_rgb_encoding_config2["n_features_per_level"]
        self.feature_num6 = self.ngp_rgb_encoding_config3["n_levels"] * self.ngp_rgb_encoding_config3["n_features_per_level"]
        self.app_extra_mlp = [MLPRender_PE_Res(2+self.feature_num4,48).to(device),MLPRender_PE_Res(2+self.feature_num5,48).to(device),MLPRender_PE_Res(2+self.feature_num6,48).to(device)]
        super(TensorVMSplit, self).__init__(aabb, gridSize, device, **kargs)
        
    def init_svd_volume(self, res, device):
        self.density_plane, self.density_line = self.init_one_svd([16,16,16], self.gridSize, 0.1, device)
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)


    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)
    
    

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001,iters = 0):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz}, 
                     #{'params': self.app_plane, 'lr': lr_init_spatialxyz},
                         {'params': self.basis_mat.parameters(), 'lr':lr_init_network},
                         {"params":self.ngp_density_encoding[0].parameters(),"lr":lr_init_spatialxyz},
                         {"params":self.ngp_density_encoding[1].parameters(),"lr":lr_init_spatialxyz},
                         {"params":self.ngp_density_encoding[2].parameters(),"lr":lr_init_spatialxyz},
                         {"params":self.ngp_rgb_encoding[0].parameters(),"lr":lr_init_spatialxyz},
                         {"params":self.ngp_rgb_encoding[1].parameters(),"lr":lr_init_spatialxyz},
                         {"params":self.ngp_rgb_encoding[2].parameters(),"lr":lr_init_spatialxyz},
                        #  {"params":self.density_network[0].parameters(),"lr":0.001},
                        #  {"params":self.density_network[1].parameters(),"lr":0.001},
                        #  {"params":self.density_network[2].parameters(),"lr":0.001},
                        #  {"params":self.extra_mlp[0].parameters(),"lr":0.001},
                        #  {"params":self.extra_mlp[1].parameters(),"lr":0.001},
                        #  {"params":self.extra_mlp[2].parameters(),"lr":0.001}
                    ]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        if iters >= 7000:
            grad_vars += [
                         {"params":self.extra_mlp[0].parameters(),"lr":0.001},
                         {"params":self.extra_mlp[1].parameters(),"lr":0.001},
                         {"params":self.extra_mlp[2].parameters(),"lr":0.001},
                         {"params":self.app_extra_mlp[0].parameters(),"lr":0.001},
                         {"params":self.app_extra_mlp[1].parameters(),"lr":0.001},
                         {"params":self.app_extra_mlp[2].parameters(),"lr":0.001}
                ]
        return grad_vars


    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]
            
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)
    
    def density_L1(self):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + torch.mean(torch.abs(self.ngp_density_encoding[idx].params)) + torch.mean(torch.abs(self.density_line[idx]))# + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        return total
    
    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.ngp_density_encoding[idx].params) * 1e-2 #+ reg(self.density_line[idx]) * 1e-3
        return total
        
    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2 #+ reg(self.app_line[idx]) * 1e-3
        return total

    def compute_densityfeature(self, xyz_sampled):
        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]],
                                        xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack(
            (xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1,
                                                                                                                  1, 2)
        ngp_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.density_line)):
            # temp = coordinate_plane[[idx_plane]]
            # plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]], #self.density_plane[idx_plane] [1,16,128,128]
            #                                  align_corners=True).view(-1, *xyz_sampled.shape[:1])
            # line_coef_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
            #                                 align_corners=True).view(-1, *xyz_sampled.shape[:1])
            ngp_line_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]], # [16,N]
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
            # sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

            normed_coordinate = (coordinate_plane[[idx_plane]].view(-1, 2) + 1)/2 # normalized from [-1,1] to [0,1]
            # normed_coordinate = coordinate_plane[[idx_plane]].view(-1, 2)
            ngp_out = self.ngp_density_encoding[idx_plane](normed_coordinate) # [N,16]
            #ngp_out = self.density_network[idx_plane](ngp_out)
            #ngp_out = self.extra_mlp[idx_plane](normed_coordinate,ngp_out)
            #Each plane needs a unique hashtable to store its feature!!!
            #print(ngp_out.shape,ngp_line_point.shape)
            ngp_feature = ngp_feature + torch.sum(ngp_out.T * ngp_line_point,dim=0)
            '''
            can we just use the concat way to combine the feature and decode it by a MLP?
            the MLP parameters is tiny,but the performance can be much better!
            '''
            #ngp_feature.append(ngp_out * ngp_line_point.T)

            #ngp_feature = torch.cat(ngp_feature,dim=1)
            sigma_feature = None


        return sigma_feature,ngp_feature



    def compute_appfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        rgb_feature = []
        for idx_plane in range(len(self.app_plane)):
            # plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
            #                                     align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            normed_coordinate = (coordinate_plane[[idx_plane]].view(-1, 2) + 1)/2
            rgb_out = self.ngp_rgb_encoding[idx_plane](normed_coordinate)
            rgb_out = self.app_extra_mlp[idx_plane](normed_coordinate,rgb_out)
            rgb_feature.append(rgb_out.T)
            #print(rgb_out.shape) # [N,48]
            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            #print(line_coef_point[0].shape) # [48,N]
        rgb_feature, line_coef_point = torch.cat(rgb_feature), torch.cat(line_coef_point)


        return self.basis_mat((rgb_feature * line_coef_point).T)



    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))


        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        # print(new_aabb, self.aabb)
        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            mode0, mode1 = self.matMode[i]
            # self.density_plane[i] = torch.nn.Parameter(
            #     self.density_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            # )
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )


        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))


