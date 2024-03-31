import torch
import torch.nn as nn
from .tensorBase import *
import tinycudann as tcnn



class ProgressiveBandHashGrid(nn.Module):
    def __init__(self, in_channels,  encoding_config, final_reso, dtype=torch.float32,device = "cuda:0"):
        super().__init__()
        self.n_input_dims = in_channels
        self.max_level = 5
        self.encodings = []
        with torch.cuda.device(device):
            self.Encoding = tcnn.Encoding(in_channels, encoding_config, dtype = dtype)


        self.n_features_per_levels = []
        self.feature_nums = []
        self.mask_length = 0
        self.n_features_per_levels = encoding_config["n_features_per_level"]
        self.feature_nums = encoding_config["n_levels"] * self.n_features_per_levels
        config = {
            "start_level":1,
            "start_step":0,
            "update_steps":3000,
        } # write this config in config file can be better
        self.start_level, self.start_step, self.update_steps = (
            config["start_level"],
            config["start_step"],
            config["update_steps"],
        )
        self.current_level = self.start_level
        # self.mask = torch.zeros(
        #     self.mask_length,
        #     dtype=torch.float32,
        #     device=device,
        # )
        # self.flag = [1,0,0,0,0]

    def forward(self, x):
        enc = self.Encoding(x)
        return enc

    def update_step(self, global_step):
        current_level = min(
            self.start_level
            + max(global_step - self.start_step, 0) // self.update_steps,
            self.max_level,
        )
        if current_level > self.current_level:
            print(f"Update current level to {current_level}")
        self.current_level = current_level

        #transplant the parameters to the next level
        last_params = dict(self.Encoding.named_parameters())


        # index_query = {1: self.feature_num_1,2:self.feature_num_1+self.feature_num_2,3:self.feature_num_1+self.feature_num_2+self.feature_num_3}
        # index = index_query[self.current_level]
        # self.mask[: index] = 1.0



class TensorVMSplitWithNGP(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        self.density_encoding_config = [{},{},{}]
        self.density_encoding_config[0] = {
            "n_levels": 1,
            "otype": "DenseGrid",
            "n_features_per_level": 16,
            "base_resolution": 300,
            "log2_hashmap_size": 17, #2^16 = 65536
            "per_level_scale": 1.5
        }
        self.density_encoding_config[1] = {
            "n_levels": 1,
            "otype": "DenseGrid",
            "n_features_per_level": 16,
            "base_resolution": 300,
            "log2_hashmap_size": 17,
            "per_level_scale": 1.5
        }
        self.density_encoding_config[2] = {
            "n_levels": 1,
            "otype": "DenseGrid",
            "n_features_per_level": 16,
            "base_resolution": 300,
            "log2_hashmap_size": 17,
            "per_level_scale": 1.5
        }
        self.app_encoding_config_1 = {
            "n_levels": 1,
            "otype": "HashGrid",
            "n_features_per_level": 32,
            "base_resolution": 300,
            "log2_hashmap_size": 17,
            "per_level_scale": 1
        }
        self.app_encoding_config_2 = {
            "n_levels": 1,
            "otype": "HashGrid",
            "n_features_per_level": 32,
            "base_resolution": 300,
            "log2_hashmap_size": 17,
            "per_level_scale": 1
        }
        self.app_encoding_config_3 = {
            "n_levels": 1,
            "otype": "HashGrid",
            "n_features_per_level": 32,
            "base_resolution": 300,
            "log2_hashmap_size": 17,
            "per_level_scale": 1
        }
        super(TensorVMSplitWithNGP, self).__init__(aabb, gridSize, device, **kargs)
        self.dtype=torch.float32
        self.final_reso_1 = 300
        self.final_reso_2 = 300
        self.final_reso_3 = 300

        # self.ngp_density_encoding = [ProgressiveBandHashGrid(2,self.density_encoding_config_1,self.final_reso_1),
        #                              ProgressiveBandHashGrid(2,self.density_encoding_config_2,self.final_reso_2),
        #                              ProgressiveBandHashGrid(2,self.density_encoding_config_3,self.final_reso_3)]
        self.ngp_density_encoding = [tcnn.Encoding(2,self.density_encoding_config[0],dtype=self.dtype),
                                     tcnn.Encoding(2,self.density_encoding_config[1],dtype=self.dtype),
                                     tcnn.Encoding(2,self.density_encoding_config[2],dtype=self.dtype)]
        self.ngp_app_encoding = [tcnn.Encoding(2,self.app_encoding_config_1,dtype=self.dtype),
                                 tcnn.Encoding(2,self.app_encoding_config_2,dtype=self.dtype),
                                 tcnn.Encoding(2,self.app_encoding_config_3,dtype=self.dtype)]

    def HashEncoding(self,feature_plane,coordinate_plane):
        '''
        usage: use a hashtable to store and compact the feature

        '''
    def init_svd_volume(self, res, device):
        #self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)
        # NOTE: use the same way as progresssive plane encoding
        f_x = self.density_encoding_config[0]["n_levels"] * self.density_encoding_config[0]["n_features_per_level"]
        f_y = self.density_encoding_config[1]["n_levels"] * self.density_encoding_config[1]["n_features_per_level"]
        f_z = self.density_encoding_config[2]["n_levels"] * self.density_encoding_config[2]["n_features_per_level"]
        #self.ngp_plane, self.ngp_line = self.init_one_svd([feature_num,feature_num,feature_num], self.gridSize, 0.1, device)
        self.ngp_line = self.init_one_line([f_x,f_y,f_z], self.gridSize, 0.1, device)
        app_f_x = self.app_encoding_config_1["n_levels"] * self.app_encoding_config_1["n_features_per_level"]
        app_f_y = self.app_encoding_config_2["n_levels"] * self.app_encoding_config_2["n_features_per_level"]
        app_f_z = self.app_encoding_config_3["n_levels"] * self.app_encoding_config_3["n_features_per_level"]
        self.ngp_app_line = self.init_one_line([app_f_x,app_f_y,app_f_z], self.gridSize,0.1,device)
        #NGP_APP_FEATURE'S  basis_mat
        #self.basis_mat = torch.nn.Linear(app_f_x+app_f_y+app_f_z, self.app_dim, bias=False).to(device)
        #original basis mat
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  # feature plane
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))  # feature line

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

    def init_one_line(self, n_component, GridSize, scale, device):
        line = []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            line.append(torch.nn.Parameter(scale * torch.randn((1, n_component[i], GridSize[vec_id], 1))))

        return torch.nn.ParameterList(line).to(device)

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001,lr_ngp=0.02):
        # ngp_app and OG_density
        grad_vars = [
                     # {'params': self.density_line, 'lr': lr_init_spatialxyz},
                     # {'params': self.density_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz},
                     {'params': self.app_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(), 'lr': lr_init_network},
                     # {'params': self.ngp_app_encoding[0].parameters(), 'lr': lr_ngp},
                     # {'params': self.ngp_app_encoding[1].parameters(), 'lr': lr_ngp},
                     # {'params': self.ngp_app_encoding[2].parameters(), 'lr': lr_ngp},
                     # {'params': self.ngp_app_encoding[0].parameters(), 'lr': lr_ngp},
                     # {'params': self.ngp_app_encoding[1].parameters(), 'lr': lr_ngp},
                     # {'params': self.ngp_app_encoding[2].parameters(), 'lr': lr_ngp},
                     {'params': self.ngp_density_encoding[0].parameters(), 'lr': lr_ngp},
                     {'params': self.ngp_density_encoding[1].parameters(), 'lr': lr_ngp},
                     {'params': self.ngp_density_encoding[2].parameters(), 'lr': lr_ngp},
                     {'params': self.ngp_line.parameters(), 'lr': lr_ngp},
                     # {'params': self.ngp_app_line.parameters(), 'lr': lr_ngp},
                     ]
        # grid_vars2 = [
        #              {'params': self.ngp_app_encoding[0].encoding_two.parameters(), 'lr': lr_ngp},
        #              {'params': self.ngp_app_encoding[1].encoding_two.parameters(), 'lr': lr_ngp},
        #              {'params': self.ngp_app_encoding[2].encoding_two.parameters(), 'lr': lr_ngp},
        #              {'params': self.ngp_density_encoding[0].encoding_two.parameters(), 'lr': lr_ngp},
        #              {'params': self.ngp_density_encoding[1].encoding_two.parameters(), 'lr': lr_ngp},
        #              {'params': self.ngp_density_encoding[2].encoding_two.parameters(), 'lr': lr_ngp},
        #              ]
        # grid_vars3 = [
        #     {'params': self.ngp_app_encoding[0].encoding_three.parameters(), 'lr': lr_ngp},
        #     {'params': self.ngp_app_encoding[1].encoding_three.parameters(), 'lr': lr_ngp},
        #     {'params': self.ngp_app_encoding[2].encoding_three.parameters(), 'lr': lr_ngp},
        #     {'params': self.ngp_density_encoding[0].encoding_three.parameters(), 'lr': lr_ngp},
        #     {'params': self.ngp_density_encoding[1].encoding_three.parameters(), 'lr': lr_ngp},
        #     {'params': self.ngp_density_encoding[2].encoding_three.parameters(), 'lr': lr_ngp},
        # ]
        #both ngp and OG
        # grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz},
        #              {'params': self.density_plane, 'lr': lr_init_spatialxyz},
        #              {'params': self.app_line, 'lr': lr_init_spatialxyz},
        #              {'params': self.app_plane, 'lr': lr_init_spatialxyz},
        #              {'params': self.basis_mat.parameters(), 'lr': lr_init_network},
        #              {'params': self.ngp_density_encoding[0].parameters(), 'lr': lr_ngp},
        #              {'params': self.ngp_density_encoding[1].parameters(), 'lr': lr_ngp},
        #              {'params': self.ngp_density_encoding[2].parameters(), 'lr': lr_ngp},
        #              {'params': self.ngp_app_encoding.parameters(), 'lr': lr_ngp},]
        #only optimize ngp params
        # grad_vars = [
        #              {'params': self.app_line, 'lr': lr_init_spatialxyz},
        #              {'params': self.app_plane, 'lr': lr_init_spatialxyz},
        #              {'params': self.basis_mat.parameters(), 'lr': lr_init_network},
        #              {'params': self.ngp_density_encoding[0].parameters(), 'lr': lr_ngp},
        #              {'params': self.ngp_density_encoding[1].parameters(), 'lr': lr_ngp},
        #              {'params': self.ngp_density_encoding[2].parameters(), 'lr': lr_ngp},
        #              {'params': self.ngp_line.parameters(), 'lr': lr_ngp},]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params': self.renderModule.parameters(), 'lr': lr_init_network}]
        return grad_vars

    def vectorDiffs(self, vector_comps):
        total = 0

        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]

            dotp = torch.matmul(vector_comps[idx].view(n_comp, n_size),
                                vector_comps[idx].view(n_comp, n_size).transpose(-1, -2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp - 1, n_comp + 1)[..., :-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)

    def density_L1(self):
        total = 0
        for idx in range(len(self.ngp_line)):
            total = (total + torch.mean(torch.abs(self.ngp_app_encoding[idx].params))
                     + torch.mean(torch.abs(self.ngp_density_encoding[idx].params)))  # + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.ngp_line)):
            total = total + reg(self.ngp_density_encoding[idx].params) * 1e-2  # + reg(self.density_line[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.ngp_app_line)):
            total = total + reg(self.ngp_app_encoding[idx].encoding.params) * 1e-2  # + reg(self.app_line[idx]) * 1e-3
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
        for idx_plane in range(len(self.ngp_line)):
            # temp = coordinate_plane[[idx_plane]]
            # plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]], #self.density_plane[idx_plane] [1,16,128,128]
            #                                  align_corners=True).view(-1, *xyz_sampled.shape[:1])
            # line_coef_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
            #                                 align_corners=True).view(-1, *xyz_sampled.shape[:1])
            ngp_line_point = F.grid_sample(self.ngp_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
            # sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

            normed_coordinate = (coordinate_plane[[idx_plane]].view(-1, 2) + 1)/2# normalized from [-1,1] to [0,1]
            # normed_coordinate = coordinate_plane[[idx_plane]].view(-1, 2)
            ngp_out = self.ngp_density_encoding[idx_plane](normed_coordinate)
            #Each plane needs a unique hashtable to store its feature!!!
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
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]],
                                        xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack(
            (xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1,
                                                                                                                  1, 2)
        ngp_feature = []
        plane_coef_point, line_coef_point = [], []
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                  align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                                 align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)

        return self.basis_mat((plane_coef_point * line_coef_point).T)

    def compute_appfeature_withngp(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]],
                                        xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack(
            (xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1,
                                                                                                                  1, 2)
        ngp_feature = []
        ngp_line_feature = []
        plane_coef_point, line_coef_point = [], []
        for idx_plane in range(len(self.ngp_app_line)):
            #plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
            #                                      align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            #line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
            #                                     align_corners=True).view(-1, *xyz_sampled.shape[:1]))

            ngp_line_feature.append(F.grid_sample(self.ngp_app_line[idx_plane], coordinate_line[[idx_plane]],
                                                 align_corners=True).view(-1, *xyz_sampled.shape[:1]) )
            normed_coordinate = (coordinate_plane[[idx_plane]].view(-1,2) + 1)/2 # normalized from [-1,1] to [0,1]
            ngp_feature.append(self.ngp_app_encoding[idx_plane](normed_coordinate))
        #plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        ngp_feature = torch.cat(ngp_feature, dim=1)
        ngp_line_feature = torch.cat(ngp_line_feature)

        '''
        NOTE : thie part ngp feature only have plane feature but no line feature.
        Finished now
        '''

        return None, self.basis_mat(ngp_feature * ngp_line_feature.T)

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            #NOTE:Below this, I tried to modify the NGP to get bigger and bigger resolution,and failed!
            #So in this part,I just upsample the vector feature.
            #get the encoding param
            # encoding_param = dict(plane_coef[i].named_parameters())
            #
            # current_reso = self.density_encoding_config[i]["base_resolution"]

            #self.density_encoding_config[i]["base_resolution"] +=64
            #plane_coef[i] = tcnn.Encoding(2,self.density_encoding_config[i],dtype=self.dtype)
            #inter_data = encoding_param["params"].data.view(1,16,current_reso,current_reso)


            #new_param = torch.nn.Parameter(
            #    F.interpolate(inter_data, size=(current_reso+64, current_reso+64), mode='nearest',
            #                  align_corners=None))
            #plane_coef[i].load_state_dict({"params":new_param.view(-1)})
            # plane_coef[i] = torch.nn.Parameter(
            #     F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
            #                   align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        return plane_coef, line_coef

    @torch.no_grad()
    def up_sampling_Line(self, line , res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            line[i] = torch.nn.Parameter(
                F.interpolate(line[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        return line
    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        #self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        self.ngp_density_encoding, self.ngp_line = self.up_sampling_VM(self.ngp_density_encoding, self.ngp_line, res_target)
        #self.ngp_line = self.up_sampling_Line(self.ngp_line,res_target)
        self.ngp_app_line = self.up_sampling_Line(self.ngp_app_line, res_target)

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
                self.density_line[i].data[..., t_l[mode0]:b_r[mode0], :]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[..., t_l[mode0]:b_r[mode0], :]
            )
            mode0, mode1 = self.matMode[i]
            # self.density_plane[i] = torch.nn.Parameter(
            #     self.density_plane[i].data[..., t_l[mode1]:b_r[mode1], t_l[mode0]:b_r[mode0]]
            # )
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[..., t_l[mode1]:b_r[mode1], t_l[mode0]:b_r[mode0]]
            )

        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize - 1), (b_r - 1) / (self.gridSize - 1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1 - t_l_r) * self.aabb[0] + t_l_r * self.aabb[1]
            correct_aabb[1] = (1 - b_r_r) * self.aabb[0] + b_r_r * self.aabb[1]
            print("aabb", new_aabb, "correct aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

    @torch.no_grad()
    def shrink_ngp(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        # print(new_aabb, self.aabb)
        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)
        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.ngp_line[i] = torch.nn.Parameter(
                self.ngp_line[i].data[..., t_l[mode0]:b_r[mode0], :]
            )
            self.ngp_app_line[i] = torch.nn.Parameter(
                self.ngp_app_line[i].data[..., t_l[mode0]:b_r[mode0], :]
            )
            # mode0, mode1 = self.matMode[i]
            # self.density_plane[i] = torch.nn.Parameter(
            #     self.density_plane[i].data[..., t_l[mode1]:b_r[mode1], t_l[mode0]:b_r[mode0]]
            # )
            # self.app_plane[i] = torch.nn.Parameter(
            #     self.app_plane[i].data[..., t_l[mode1]:b_r[mode1], t_l[mode0]:b_r[mode0]]
            #
        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize - 1), (b_r - 1) / (self.gridSize - 1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1 - t_l_r) * self.aabb[0] + t_l_r * self.aabb[1]
            correct_aabb[1] = (1 - b_r_r) * self.aabb[0] + b_r_r * self.aabb[1]
            print("aabb", new_aabb, "correct aabb", correct_aabb)
            new_aabb = correct_aabb
        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))
