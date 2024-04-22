import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks,BasicConv2d
from ..backbones.gcn import GCN

def L_Matrix(adj_npy, adj_size):
    D = np.zeros((adj_size, adj_size))
    for i in range(adj_size):
        tmp = adj_npy[i, :]
        count = np.sum(tmp == 1)
        if count > 0:
            number = count ** (-1 / 2)
            D[i, i] = number

    x = np.matmul(D, adj_npy)
    L = np.matmul(x, D)
    return L


def get_fine_adj_npy():
    fine_adj_list = [
        # 1  2  3  4  5  6  7  8  9  10 11
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 1
        [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0],  # 2
        [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1],  # 3
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1],  # 4
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # 5
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # 6
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1],  # 7
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1],  # 8
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],  # 9
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # 10
        [1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]  # 11
    ]
    fine_adj_npy = np.array(fine_adj_list)
    fine_adj_npy = L_Matrix(fine_adj_npy, len(fine_adj_npy))  # len返回的是行数
    return fine_adj_npy


def get_coarse_adj_npy():
    coarse_adj_list = [
        # 1  2  3  4  5
        [1, 1, 1, 1, 1],  # 1
        [1, 1, 0, 0, 0],  # 2
        [1, 0, 1, 0, 0],  # 3
        [1, 0, 0, 1, 0],  # 4
        [1, 0, 0, 0, 1]  # 5
    ]
    coarse_adj_npy = np.array(coarse_adj_list)
    coarse_adj_npy = L_Matrix(coarse_adj_npy, len(coarse_adj_npy))  # len返回的是行数
    return coarse_adj_npy


class Msgg_ParsingGait(BaseModel):
    def __init__(self, cfgs, training):
        super().__init__(cfgs, training)
        # self.y_sp_output = None

    def semantic_pooling(self, x):
        cur_node_num = x.size()[-1]
        half_x_1, half_x_2 = torch.split(x, int(cur_node_num / 2), dim=-1)
        x_sp = torch.add(half_x_1, half_x_2) / 2
        return x_sp
    def build_network_Msgg(self, model_cfg):
        in_c = model_cfg['in_channels']
        out_c = model_cfg['out_channels']
        num_id = model_cfg['num_id']

        temporal_kernel_size = model_cfg['temporal_kernel_size']

        # load spatial graph
        self.graph = SpatialGraph(**model_cfg['graph_cfg'])
        A_lowSemantic = torch.tensor(self.graph.get_adjacency(semantic_level=0), dtype=torch.float32, requires_grad=False)
        A_mediumSemantic =  torch.tensor(self.graph.get_adjacency(semantic_level=1), dtype=torch.float32, requires_grad=False)
        # A_highSemantic = torch.tensor(self.graph.get_adjacency(semantic_level=2), dtype=torch.float32, requires_grad=False)

        self.register_buffer('A_lowSemantic', A_lowSemantic)
        self.register_buffer('A_mediumSemantic', A_mediumSemantic)
        # self.register_buffer('A_highSemantic', A_highSemantic)

        # build networks
        spatial_kernel_size = self.graph.num_A
        temporal_kernel_size = temporal_kernel_size
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.st_gcn_networks_lowSemantic = nn.ModuleList()
        self.st_gcn_networks_mediumSemantic = nn.ModuleList()
        self.st_gcn_networks_highSemantic = nn.ModuleList()
        for i in range(len(in_c)-1):
            if i == 0:
                self.st_gcn_networks_lowSemantic.append(st_gcn_block(in_c[i], in_c[i+1], kernel_size, 1, residual=False))
                self.st_gcn_networks_mediumSemantic.append(st_gcn_block(in_c[i], in_c[i+1], kernel_size, 1, residual=False))
                # self.st_gcn_networks_highSemantic.append(st_gcn_block(in_c[i], in_c[i+1], kernel_size, 1, residual=False))
            else:
                self.st_gcn_networks_lowSemantic.append(st_gcn_block(in_c[i], in_c[i+1], kernel_size, 1))
                self.st_gcn_networks_mediumSemantic.append(st_gcn_block(in_c[i], in_c[i+1], kernel_size, 1))
                # self.st_gcn_networks_highSemantic.append(st_gcn_block(in_c[i], in_c[i+1], kernel_size, 1))

            self.st_gcn_networks_lowSemantic.append(st_gcn_block(in_c[i+1], in_c[i+1], kernel_size, 1))
            self.st_gcn_networks_mediumSemantic.append(st_gcn_block(in_c[i+1], in_c[i+1], kernel_size, 1))
            # self.st_gcn_networks_highSemantic.append(st_gcn_block(in_c[i+1], in_c[i+1], kernel_size, 1))
        #     这里设置成parameter认为是一个可以更新的权重参数，并且采用torch.ones方式其实是使用了类似注意力的机制
        self.edge_importance_lowSemantic = nn.ParameterList([
            nn.Parameter(torch.ones(self.A_lowSemantic.size()))
            for i in self.st_gcn_networks_lowSemantic])

        self.edge_importance_mediumSemantic = nn.ParameterList([
            nn.Parameter(torch.ones(self.A_mediumSemantic.size()))
            for i in self.st_gcn_networks_mediumSemantic])

        # self.edge_importance_highSemantic = nn.ParameterList([
        #     nn.Parameter(torch.ones(self.A_highSemantic.size()))
        #     for i in self.st_gcn_networks_highSemantic])

        self.fc = nn.Linear(in_c[-1], out_c)
        self.bn_neck = nn.BatchNorm1d(out_c)
        self.encoder_cls = nn.Linear(out_c, num_id, bias=False)
    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        # self.Backbone_new = self.get_backbone(model_cfg['backbone_cfg_new'])
        # self.Backbone_new = SetBlockWrapper(self.Backbone_new)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.TP_new = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])
        self.conv2 = BasicConv2d(model_cfg['CNN']['in_channels'], model_cfg['CNN']['out_channels'], 3, 1, 1)
        nfeat = model_cfg['GCN']['in_channels']
        nfeat_new= model_cfg['SeparateFCs']['in_channels']
        gcn_cfg = model_cfg['gcn_cfg']
        self.fine_parts = gcn_cfg['fine_parts']
        coarse_parts = gcn_cfg['coarse_parts']

        self.only_fine_graph = gcn_cfg['only_fine_graph']
        self.only_coarse_graph = gcn_cfg['only_coarse_graph']
        self.combine_fine_coarse_graph = gcn_cfg['combine_fine_coarse_graph']

        if self.only_fine_graph:
            fine_adj_npy = get_fine_adj_npy()
            self.fine_adj_npy = torch.from_numpy(fine_adj_npy).float()
            self.gcn_fine = GCN(self.fine_parts, nfeat, nfeat, isMeanPooling=True)
            self.gammas_fine = torch.nn.Parameter(torch.ones(self.fine_parts) * 0.75)
        elif self.only_coarse_graph:
            coarse_adj_npy = get_coarse_adj_npy()
            self.coarse_adj_npy = torch.from_numpy(coarse_adj_npy).float()
            self.gcn_coarse = GCN(coarse_parts, nfeat, nfeat, isMeanPooling=True)
            self.gcn_coarse_new = GCN(coarse_parts, nfeat_new, nfeat_new, isMeanPooling=True)
            self.gammas_coarse = torch.nn.Parameter(torch.ones(coarse_parts) * 0.75)
        elif self.combine_fine_coarse_graph:
            fine_adj_npy = get_fine_adj_npy()
            self.fine_adj_npy = torch.from_numpy(fine_adj_npy).float()
            self.gcn_fine = GCN(self.fine_parts, nfeat, nfeat, isMeanPooling=True)
            self.gammas_fine = torch.nn.Parameter(torch.ones(self.fine_parts) * 0.75)
            coarse_adj_npy = get_coarse_adj_npy()
            self.coarse_adj_npy = torch.from_numpy(coarse_adj_npy).float()
            self.gcn_coarse = GCN(coarse_parts, nfeat, nfeat, isMeanPooling=True)
            self.gammas_coarse = torch.nn.Parameter(torch.ones(coarse_parts) * 0.75)
        else:
            raise ValueError("You should choose fine/coarse graph, or combine both of them.")
        # msgg模块build
        self.build_network_Msgg(model_cfg)

    def PPforGCN(self, x):
        """
            Part Pooling for GCN
            x   : [n, p, c, h, w]
            ret : [n, p, c]
        """
        n, p, c, h, w = x.size()
        z = x.view(n, p, c, -1)  # [n, p, c, h*w]
        z = z.mean(-1) + z.max(-1)[0]  # [n, p, c]
        return z

    def ParsPartforFineGraph(self, mask_resize, z):
        """
            x: [n, c, s, h, w]
            paes: [n, 1, s, H, W]
            return [n*s, 11, c, h, w]
            ***Fine Parts:
            # 0: Background,
            1: Head,
            2: Torso,
            3: Left-arm,
            4: Right-arm,
            5: Left-hand,
            6: Right-hand,
            7: Left-leg,
            8: Right-leg,
            9: Left-foot,
            10: Right-foot,
            11: Dress
        """
        fine_mask_list = list()
        for i in range(1, self.fine_parts + 1):
            fine_mask_list.append((mask_resize.long() == i))  # split mask of each class

        fine_z_list = list()
        for i in range(len(fine_mask_list)):
            mask = fine_mask_list[i].unsqueeze(1)
            fine_z_list.append(
                (mask.float() * z * self.gammas_fine[i] + (~mask).float() * z * (1.0 - self.gammas_fine[i])).unsqueeze(
                    1))  # split feature map by mask of each class
        fine_z_feat = torch.cat(fine_z_list, dim=1)  # [n*s, 11, c, h, w] or [n*s, 5, c, h, w]

        return fine_z_feat

    def ParsPartforCoarseGraph(self, mask_resize, z):
        """
            x: [n, c, s, h, w]
            paes: [n, 1, s, H, W]
            return [n*s, 5, c, h, w]
            ***Coarse Parts:
            1: [1, 2, 11]  Head, Torso, Dress
            2: [3, 5]  Left-arm, Left-hand
            3: [4, 6]  Right-arm, Right-hand
            4: [7, 9]  Left-leg, Left-foot
            5: [8, 10] Right-leg, Right-foot
        """
        coarse_mask_list = list()
        coarse_parts = [[1, 2, 11], [3, 5], [4, 6], [7, 9], [8, 10]]
        for coarse_part in coarse_parts:
            # 这里获取背景部分
            part = mask_resize.long() == -1
            for i in coarse_part:
                # 把各个label进行相加， 组成一个mask
                part += (mask_resize.long() == i)
            coarse_mask_list.append(part)

        coarse_z_list = list()
        for i in range(len(coarse_mask_list)):
            # 这里把原来mask 维度 nxs,h,w变成nxs,1,h,w
            mask = coarse_mask_list[i].unsqueeze(1)
            # 这里z是nxs,c,h,w，  要进行mask就需要对每个feature的所有通道进行相同的mask，所以是1个mask对应一个c通道
            coarse_z_list.append((mask.float() * z * self.gammas_coarse[i] + (~mask).float() * z * (
                        1.0 - self.gammas_coarse[i])).unsqueeze(1))  # split feature map by mask of each class

        coarse_z_feat = torch.cat(coarse_z_list, dim=1)  # [n*s, 11, c, h, w] or [n*s, 5, c, h, w]

        return coarse_z_feat

    def ParsPartforGCN(self, x, pars):
        """
            x: [n, c, s, h, w]  output by CNN
            pars: [n, 1, s, H, W]:input parsing
            return [n*s, 11, c, h, w] or [n*s, 5, c, h, w]
        """
        n, c, s, h, w = x.size()
        # mask_resize: [n, s, h, w]
        mask_resize = F.interpolate(input=pars.squeeze(1), size=(h, w), mode='nearest')
        mask_resize = mask_resize.view(n * s, h, w)

        z = x.transpose(1, 2).reshape(n * s, c, h, w)

        if self.only_fine_graph:
            fine_z_feat = self.ParsPartforFineGraph(mask_resize, z)
            return fine_z_feat, None
        elif self.only_coarse_graph:
            coarse_z_feat = self.ParsPartforCoarseGraph(mask_resize, z)
            return None, coarse_z_feat
        elif self.combine_fine_coarse_graph:
            fine_z_feat = self.ParsPartforFineGraph(mask_resize, z)
            coarse_z_feat = self.ParsPartforCoarseGraph(mask_resize, z)
            return fine_z_feat, coarse_z_feat
        else:
            raise ValueError("You should choose fine/coarse graph, or combine both of them.")

    def get_gcn_feat(self, n, input, adj_np, is_cuda, seqL):
        input_ps = self.PPforGCN(input)  # [n*s, 11, c]
        n_s, p, c = input_ps.size()
        # print("GHGHGH  ",input_ps.size())
        if is_cuda:
            adj = adj_np.cuda()
        adj = adj.repeat(n_s, 1, 1)
        if p == 11:
            output_ps = self.gcn_fine(input_ps, adj)  # [n*s, 11, c]
        elif p == 5:
            output_ps = self.gcn_coarse(input_ps, adj)  # [n*s, 5, c]
        else:
            raise ValueError(f"The parsing parts should be 11 or 5, but got {p}")
        # output_ps = output_ps.view(n, n_s // n, p, c)  # [n, s, ps, c]
        # 这里暂时不进行序列的时间池化
        # output_ps = self.TP(output_ps, seqL, dim=1, options={"dim": 1})[0]  # [n, ps, c]

        return output_ps

    def forward_msgg(self, inputs):
        ipts, labs, _, _, seqL = inputs

        # x = ipts[0]  # [N, T, V, C]
        # 获取骨骼数据
        x=ipts[1]
        del ipts
        """
           N - the number of videos.
           T - the number of frames in one video.
           V - the number of keypoints.
           C - the number of features for one keypoint.
        """
        N, T, V, C = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N, C, T, V)
        # print("****************",x.size())
        y = self.semantic_pooling(x)
        # print("((((((((((((((",y.size())
        # z = self.semantic_pooling(y)
        for gcn_lowSemantic, importance_lowSemantic, gcn_mediumSemantic, importance_mediumSemantic, gcn_highSemantic, importance_highSemantic in zip(
                self.st_gcn_networks_lowSemantic, self.edge_importance_lowSemantic, self.st_gcn_networks_mediumSemantic,
                self.edge_importance_mediumSemantic):
            x, _ = gcn_lowSemantic(x, self.A_lowSemantic * importance_lowSemantic)
            y, _ = gcn_mediumSemantic(y, self.A_mediumSemantic * importance_mediumSemantic)
            # z, _ = gcn_highSemantic(z, self.A_highSemantic * importance_highSemantic)

            # Cross-scale Message Passing
            x_sp = self.semantic_pooling(x)
            y = torch.add(y, x_sp)
            y_sp = self.semantic_pooling(y)
            # z = torch.add(z, y_sp)
        # print("$$$$$$$$$$$$$",y.size())
        # self.y_sp_output = y
        # global pooling for each layer
        x_sp = F.avg_pool2d(x, x.size()[2:])
        N, C, T, V = x_sp.size()
        x_sp = x_sp.view(N, C, T * V).contiguous()
        # self.x_sp=x_sp
        y_sp = F.avg_pool2d(y, y.size()[2:])
        N, C, T, V = y_sp.size()
        y_sp = y_sp.view(N, C, T * V).contiguous()
        # self.y_sp=y_sp
        # z = F.avg_pool2d(z, z.size()[2:])
        # N, C, T, V = z.size()
        # z = z.permute(0, 2, 3, 1).contiguous()
        # z = z.view(N, T * V, C)

        # z_fc = self.fc(z.view(N, -1))
        # bn_z_fc = self.bn_neck(z_fc)
        # z_cls_score = self.encoder_cls(bn_z_fc)

        # z_fc = z_fc.unsqueeze(-1).contiguous()  # [n, c, p]
        # self.z_fc=z_fc
        # z_cls_score = z_cls_score.unsqueeze(-1).contiguous()  # [n, c, p]
        return y, x_sp, y_sp
        # retval = {
        #     'training_feat': {
        #         'triplet_joints': {'embeddings': x_sp, 'labels': labs},
        #         'triplet_limbs': {'embeddings': y_sp, 'labels': labs},
        #         'triplet_bodyparts': {'embeddings': z_fc, 'labels': labs},
        #         'softmax': {'logits': z_cls_score, 'labels': labs}
        #     },
        #     'visual_summary': {},
        #     'inference_feat': {
        #         'embeddings': z_fc
        #     }
        # }
    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        pars = ipts[0]
        # print("))))))))))))) ",ipts[0].size(),ipts[1].size())
        # 增加骨骼数据
        if len(pars.size()) == 4:
            pars = pars.unsqueeze(1)

        del ipts
        # print("pars:",pars.size())
        outs = self.Backbone(pars)  # [n, c, s, h, w]

        outs_n, outs_c, outs_s, outs_h, outs_w = outs.size()

        # split features by parsing classes
        # outs_ps_fine: [n*s, 11, c, h, w]
        # outs_ps_coarse: [n*s, 5, c, h, w]
        outs_ps_fine, outs_ps_coarse = self.ParsPartforGCN(outs, pars)

        is_cuda = pars.is_cuda
        if self.only_fine_graph:
            outs_ps = self.get_gcn_feat(outs_n, outs_ps_fine, self.fine_adj_npy, is_cuda, seqL)  # [n, 11, c]
        elif self.only_coarse_graph:
            # 进行空间卷积和时间池化
            # 这里[n*s, ps, c]
            outs_ps = self.get_gcn_feat(outs_n, outs_ps_coarse, self.coarse_adj_npy, is_cuda, seqL)  # [n, 5, c]
        elif self.combine_fine_coarse_graph:
            outs_fine = self.get_gcn_feat(outs_n, outs_ps_fine, self.fine_adj_npy, is_cuda, seqL)  # [n, 11, c]
            outs_coarse = self.get_gcn_feat(outs_n, outs_ps_coarse, self.coarse_adj_npy, is_cuda, seqL)  # [n, 5, c]
            outs_ps = torch.cat([outs_fine, outs_coarse], 1)  # [n, 16, c]
        else:
            raise ValueError("You should choose fine/coarse graph, or combine both of them.")
        # outs_ps = outs_ps.transpose(1, 2).contiguous()  # [n, c, ps]

        # Temporal Pooling, TP
        # outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        # feat = self.HPP(outs)  # [n, c, p ]
        # 调用msgg模块
        y, x_sp, y_sp=self.forward_msgg(inputs)
        # Concatenate the features
        # y_sp(N, C, T, V)
        # outs_ps [n, t, v, c]
        # 这里保证两者的维度是一样的
        y_sp_output=y.permute(0, 2, 3, 1).contiguous()
        # 32,30,6,128
        n, t, v, c = y_sp_output.size()
        y_sp_output=y_sp_output.view(n*t, v, c)
        # print("^^^^^^^^^",y_sp_output.size())
        # [0,4,1,5,2]
        # [0-shoulder,1-right_elbow_wrist,2-right_knee_ankle,3-right_left_hip,4-left_elbow_wrist,5-left_knee_ankle]
        #   1: [1, 2, 11]  Head, Torso, Dress
        #             2: [3, 5]  Left-arm, Left-hand
        #             3: [4, 6]  Right-arm, Right-hand
        #             4: [7, 9]  Left-leg, Left-foot
        #             5: [8, 10] Right-leg, Right-foot
        # 对应关系，
        # y_sp_output 的节点数进行减少，把0和3节点合并取平均值
        avg_node= torch.mean(y_sp_output[:, [0, 3], :], dim=1)
        # new_v_order = torch.tensor([1, 2, 3, 4])
        rank = torch.distributed.get_rank()
        Y = torch.zeros(y_sp_output.size(0), y_sp_output.size(1) - 1, y_sp_output.size(2)).half().to(f"cuda:{rank}")

        # 这里按照 [0,4,1,5,2]进行节点顺序重新排列
        # print("**************",Y.size(),y_sp_output.size())
        Y[:, [1, 2, 3, 4], :] = y_sp_output[:,[4, 1, 5, 2], :]
        Y[:, 0, :] = avg_node
        feat_msgg_parsinggait = torch.cat([Y, outs_ps], dim=-1)  # [n*s,v-1,c+C]
        # 这里把全局的shape信息融合进去
        # 这里对outs再卷积一次，保证其通道数是640  ，
        # print("******************1", outs.size())torch.Size([32, 512, 30, 16, 11])
        n1,c1,s1,h1,w1=outs.size()
        outs=outs.permute(0, 2, 1, 3,4).contiguous()
        # print("******************2", outs.size())  torch.Size([32, 30, 512, 16, 11])
        outs=outs.view(n1*s1,c1,h1,w1)
        # print("******************3", outs.size())torch.Size([960, 512, 16, 11])
        outs = self.conv2(outs)  # [n, c, s, h, w]
        # print("******************4",outs.size())  #torch.Size([960, 640, 16, 11])
        # 这里进行时序池化操作
        n1, c1, h1, w1 = outs.size()
        outs = outs.view(n1//s1,s1 , c1, h1, w1)
        outs=outs.permute(0,2,1,3,4)
        # print("***************5",outs.size())
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w] [32,640,16,11]
        # Horizontal Pooling Matching, HPM
        # print("outs", outs_ps.size())
        feat = self.HPP(outs)  # [n, c, p]  ^^^^^ torch.Size([32, 512, 16])
        # print("^^^^^^^^^^",feat.size())
        # 进行GCN操作
        n_s,v,c = feat_msgg_parsinggait.size()
        if is_cuda:
            adj = self.coarse_adj_npy.cuda()
        adj = adj.repeat(n_s, 1, 1)
        # print("***********",feat_msgg_parsinggait.size(),adj.size())
        output_ps = self.gcn_coarse_new(feat_msgg_parsinggait, adj)  # [n*s, 5, c+C]
        output_ps = output_ps.view(outs_n, n_s // outs_n, v, -1)  # [n, s, ps, c]
        # 进行时间池化
        output_ps = self.TP_new(output_ps, seqL, dim=1, options={"dim": 1})[0]  # [n, ps, c]
        # 【32,5,640】
        output_ps=output_ps.transpose(1, 2).contiguous()   # [n, c, ps]
        # 这里在进行cat操作
        feat = torch.cat([output_ps, feat], dim=-1)  # [n,c+C,v+V]
        # print("  ***********feat",feat.size())
        # feat = torch.cat([feat, outs_ps], dim=-1)  # [n, c, p+ps]
        # embed_1 = self.FCs(feat)  # [n, c, p+ps]
        embed_1 = self.FCs(feat)  # [n, c+C, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c+C, ps]
        embed = embed_1

        n, _, s, h, w = pars.size()
        retval = {
            'training_feat': {
                'triplet_joints': {'embeddings': x_sp, 'labels': labs},
                'triplet_limbs': {'embeddings': y_sp, 'labels': labs},
                # 'triplet_bodyparts': {'embeddings': z_fc, 'labels': labs},
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/pars': pars.view(n * s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval




class st_gcn_block(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size, i.e. the number of videos.
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`.
            :math:`T_{in}/T_{out}` is a length of input/output sequence, i.e. the number of frames in a video.
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = SCN(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A

class SCN(nn.Module):
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()
        # The defined module SCN are responsible only for the Spacial Graph (i.e. the graph in on frame),
        # and the parameter t_kernel_size in this situation is always set to 1.

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels,
                              out_channels * kernel_size,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)
        """
        The 1x1 conv operation here stands for the weight metrix W.
        The kernel_size here stands for the number of different adjacency matrix, 
            which are defined according to the partitioning strategy.
        Because for neighbor nodes in the same subset (in one adjacency matrix), the weights are shared. 
        It is reasonable to apply 1x1 conv as the implementation of weight function.
        """


    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A

class SpatialGraph():
    """ Use skeleton sequences extracted by Openpose/HRNet to construct Spatial-Temporal Graph

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration Partitioning
        - gait_temporal: Gait Temporal Configuration Partitioning
            For more information, please refer to the section 'Partition Strategies' in PGG.
        layout (string): must be one of the follow candidates
        - body_12: Is consists of 12 joints.
            (right shoulder, right elbow, right knee, right hip, left elbow, left knee,
             left shoulder, right wrist, right ankle, left hip, left wrist, left ankle).
            For more information, please refer to the section 'Data Processing' in PGG.
        max_hop (int): the maximal distance between two connected nodes # 1-neighbor
        dilation (int): controls the spacing between the kernel points
    """
    def __init__(self,
                 layout='body_12', # Openpose here represents for body_12
                 strategy='spatial',
                 semantic_level=0,
                 max_hop=1,
                 dilation=1):
        self.layout = layout
        self.strategy = strategy
        self.max_hop = max_hop
        self.dilation = dilation
        self.num_node, self.neighbor_link_dic = self.get_layout_info(layout)
        self.num_A = self.get_A_num(strategy)

    def __str__(self):
        return self.A

    def get_A_num(self, strategy):
        if self.strategy == 'uniform':
            return 1
        elif self.strategy == 'distance':
            return 2
        elif (self.strategy == 'spatial') or (self.strategy == 'gait_temporal'):
            return 3
        else:
            raise ValueError("Do Not Exist This Strategy")

    def get_layout_info(self, layout):
        if layout == 'body_12':
            num_node = 12
            neighbor_link_dic = {
                0: [(7, 1), (1, 0), (10, 4), (4, 6),
                     (8, 2), (2, 3), (11, 5), (5, 9),
                     (9, 3), (3, 0), (9, 6), (6, 0)],
                1: [(1, 0), (4, 0), (0, 3), (2, 3), (5, 3)],
                2: [(1, 0), (2, 0)]
            }
            return num_node, neighbor_link_dic
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_edge(self, semantic_level):
        # edge is a list of [child, parent] pairs, regarding the center node as root node
        self_link = [(i, i) for i in range(int(self.num_node / (2 ** semantic_level)))]
        neighbor_link = self.neighbor_link_dic[semantic_level]
        edge = self_link + neighbor_link
        center = []
        if self.layout == 'body_12':
            if semantic_level == 0:
                center = [0, 3, 6, 9]
            elif semantic_level == 1:
                center = [0, 3]
            elif semantic_level == 2:
                center = [0]
        return edge, center

    def get_gait_temporal_partitioning(self, semantic_level):
        if semantic_level == 0:
            if self.layout == 'body_12':
                positive_node = {1, 2, 4, 5, 7, 8, 10, 11}
                negative_node = {0, 3, 6, 9}
        elif semantic_level == 1:
            if self.layout == 'body_12':
                positive_node = {1, 2, 4, 5}
                negative_node = {0, 3}
        elif semantic_level == 2:
            if self.layout == 'body_12':
                positive_node = {1, 2}
                negative_node = {0}
        return positive_node, negative_node
            
    def get_adjacency(self, semantic_level):
        edge, center = self.get_edge(semantic_level)
        num_node = int(self.num_node / (2 ** semantic_level))
        hop_dis = get_hop_distance(num_node, edge, max_hop=self.max_hop)
                
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((num_node, num_node))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1

        normalize_adjacency = normalize_digraph(adjacency)
        # normalize_adjacency = adjacency # withoutNodeNorm

        # normalize_adjacency[a][b] = x
        # when x = 0, node b has no connection with node a within valid hop.
        # when x ≠ 0, the normalized adjacency from node b to node a is x.
        # the value of x is normalized by the number of adjacent neighbor nodes around the node b.

        if self.strategy == 'uniform':
            A = np.zeros((1, num_node, num_node))
            A[0] = normalize_adjacency
            return A
        elif self.strategy == 'distance':
            A = np.zeros((len(valid_hop), num_node, num_node))
            for i, hop in enumerate(valid_hop):
                A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
            return A
        elif self.strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((num_node, num_node))
                a_close = np.zeros((num_node, num_node))
                a_further = np.zeros((num_node, num_node))
                for i in range(num_node):
                    for j in range(num_node):
                        if hop_dis[j, i] == hop:
                            j_hop_dis = min([hop_dis[j, _center] for _center in center])
                            i_hop_dis = min([hop_dis[i, _center] for _center in center])
                            if j_hop_dis == i_hop_dis:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif j_hop_dis > i_hop_dis:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
            return A
        elif self.strategy == 'gait_temporal':
            A = []
            positive_node, negative_node = self.get_gait_temporal_partitioning(semantic_level)
            for hop in valid_hop:
                a_root = np.zeros((num_node, num_node))
                a_positive = np.zeros((num_node, num_node))
                a_negative = np.zeros((num_node, num_node))
                for i in range(num_node):
                    for j in range(num_node):
                        if hop_dis[j, i] == hop:
                            if i == j:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif j in positive_node:
                                a_positive[j, i] = normalize_adjacency[j, i]
                            else:
                                a_negative[j, i] = normalize_adjacency[j, i]
                
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_negative)
                    A.append(a_positive)
            A = np.stack(A)
            return A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    # Calculate the shortest path between nodes
    # i.e. The minimum number of steps needed to walk from one node to another
    A = np.zeros((num_node, num_node)) # Ajacent Matrix
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
