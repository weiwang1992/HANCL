import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import hyperbolic as hyper
class HANCL(nn.Module):
    def __init__(self, args, n_entity, n_relation):
        super(HANCL, self).__init__()
        self._parse_args(args, n_entity, n_relation)
        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)

        # 添加Dropout层
        self.emb_dropout = nn.Dropout(args.emb_dropout)  # 嵌入层后的dropout
        self.att_dropout = nn.Dropout(args.att_dropout)  # 注意力机制中的dropout
        self.gate_dropout = nn.Dropout(args.gate_dropout)  # 门控机制中的dropout


        
        # 门控融合机制
        self.space_gate = nn.Sequential(
            nn.Linear(2 * self.dim, self.dim),
            nn.Sigmoid()
        )
        
        # 添加空间适配层
        self.hyper_adapter = nn.Linear(self.dim, self.dim)
        self.euclid_adapter = nn.Linear(self.dim, self.dim)

        # 添加层归一化
        self.layer_norm = nn.LayerNorm(self.dim)

        self.attention = nn.Sequential(
                nn.Linear(2*self.dim, self.dim, bias=False),
                nn.LeakyReLU(0),
                self.att_dropout,  # 添加dropout
                nn.Linear(self.dim, 1, bias=False),
                nn.Sigmoid(),
                )
        self.attention2 = nn.Sequential(
                nn.Linear(2*self.dim, self.dim),  # 增加输入维度
                nn.LeakyReLU(0),
                self.att_dropout,
                nn.Linear(self.dim, 1),
                nn.Tanh()
        )
                # 双曲空间曲率参数
        self.curvature = nn.Parameter(torch.tensor(1.0))
                # 双曲空间注意力 (使用距离)

        

        # parameters for kge enbedding with graph contrastive learning
        self.ssl_temp = 0.2    # for softmax
        self.kge_weight = 1e-6  # for kge_loss
        self._init_weight()

    def forward(
        self,
        items: torch.LongTensor,
        user_CF_set: list,
        item_KG_set: list,
        user_KG_set: list,
        item_CF_set: list,
    ):

        user_CF_embeddings = []
        # [batch_size, triple_set_size, dim]
        user_emb_0 = self.entity_emb(user_CF_set[0][0])
        # [batch_size, dim]
        user_intial_embedding = user_emb_0.mean(dim=1)
        user_CF_embeddings.append(user_intial_embedding)
        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_set_emb = self.entity_emb(user_CF_set[0][0])
            h_set_emb = self.emb_dropout(h_set_emb)  # 应用dropout
    
            #h_set_emb = hyper.expmap0(h_set_emb)
      
            # [batch_size, triple_set_size, dim]
            #r_emb = self.relation_emb(user_triple_set[1][i])
            
            path_emb = self.relation_emb(user_CF_set[1][0])
            path_emb = self.emb_dropout(path_emb)  # 应用dropout
            
            # [batch_size, triple_set_size, dim]
            for k in range(1, i+1):
                #path_emb += path_emb,self.relation_emb(user_CF_set[1][i])
                h_set_emb += self.entity_emb(user_CF_set[0][i])
                #h_set_emb = hyper.mobius_add(h_set_emb,self.entity_emb(user_CF_set[0][i])) 
                path_emb=torch.mul(path_emb,self.relation_emb(user_CF_set[1][i]))
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(user_CF_set[2][i])
            t_emb = self.emb_dropout(t_emb)  # 应用dropout
            # [batch_size, triple_set_size,neighbor_size, dim]
         
            # [batch_size, dim]
            user_emb_i = self.ed_knowledge_agg(h_set_emb, path_emb, t_emb)
            #user_emb_i = self._knowledge_attention(h_set_emb, path_emb, t_emb)
            user_CF_embeddings.append(user_emb_i)


        item_KG_embeddings = []
        # [batch size, dim]
        item_emb_origin = self.entity_emb(items)
        item_intial_embedding = item_emb_origin
        item_KG_embeddings.append(item_intial_embedding)
        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_set_emb = self.entity_emb(item_KG_set[0][0])
            h_set_emb = self.emb_dropout(h_set_emb)  # 应用dropout
            # [batch_size, triple_set_size, dim]
            # [batch_size, triple_set_size, dim]
            path_emb = self.relation_emb(item_KG_set[1][0])
            path_emb = self.emb_dropout(path_emb)  # 应用dropout
            # [batch_size, triple_set_size, dim]
            for k in range(1, i+1):
                h_set_emb += self.entity_emb(item_KG_set[0][i])
                #path_emb += self.relation_emb(item_KG_set[1][i])
                path_emb=torch.mul(path_emb,self.relation_emb(item_KG_set[1][i]))
                
            t_emb = self.entity_emb(item_KG_set[2][i])
            t_emb = self.emb_dropout(t_emb)  # 应用dropout
            # [batch_size, triple_set_size,neighbor_size, dim]
       
            # [batch_size, dim]
            item_emb_i = self.ed_knowledge_agg(h_set_emb, path_emb, t_emb)
            item_KG_embeddings.append(item_emb_i)


        user_KG_embeddings = []
        user_KG_embeddings_0 = self.entity_emb(user_KG_set[0][0])
        user_intial_potential_embedding = user_KG_embeddings_0.mean(dim=1)
        user_KG_embeddings.append(user_intial_potential_embedding)
        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_set_emb = self.entity_emb(user_KG_set[0][0])
            h_set_emb = self.emb_dropout(h_set_emb)  # 应用dropout
            # [batch_size, triple_set_size, dim]
            path_emb = self.relation_emb(user_KG_set[1][0])
            path_emb = self.emb_dropout(path_emb)  # 应用dropout
            # [batch_size, triple_set_size, dim]
            for k in range(1, i+1):
                h_set_emb += self.entity_emb(user_KG_set[0][i])
                #path_emb += path_emb,self.relation_emb(user_KG_set[1][i])
                path_emb=torch.mul(path_emb,self.relation_emb(user_KG_set[1][i]))
            #r_emb = self.relation_emb(user_KG_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(user_KG_set[2][i])
            t_emb = self.emb_dropout(t_emb)  # 应用dropout
            # [batch_size, triple_set_size,neighbor_size, dim]

            # [batch_size, dim]
            user_potential_emb_i = self.hyper_knowledge_agg(h_set_emb, path_emb, t_emb)
            user_KG_embeddings.append(user_potential_emb_i)


        item_CF_embeddings = []
        item_CF_embeddings_0 = self.entity_emb(item_CF_set[0][0])
        item_intial_origin_embedding = item_CF_embeddings_0.mean(dim=1)
        item_CF_embeddings.append(item_intial_origin_embedding)
        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_set_emb = self.entity_emb(item_CF_set[0][0])
            h_set_emb = self.emb_dropout(h_set_emb)  # 应用dropout
            # [batch_size, triple_set_size, dim]
            path_emb = self.relation_emb(item_CF_set[1][0])
            path_emb = self.emb_dropout(path_emb)  # 应用dropout
            # [batch_size, triple_set_size, dim]
            for k in range(1, i+1):
                h_set_emb += self.entity_emb(item_CF_set[0][i])
                #path_emb+= self.relation_emb(item_CF_set[1][i])
                path_emb=torch.mul(path_emb,self.relation_emb(item_CF_set[1][i]))
            #r_emb = self.relation_emb(item_CF_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(item_CF_set[2][i])
            t_emb = self.emb_dropout(t_emb)  # 应用dropoutuser_CF_embeddings
            # [batch_size, triple_set_size,neighbor_size, dim]
 
            # [batch_size, dim]
            item_origin_emb_i = self.hyper_knowledge_agg(h_set_emb, path_emb, t_emb)
            item_CF_embeddings.append(item_origin_emb_i)
            
        scores = self.predict(user_CF_embeddings, item_KG_embeddings, user_KG_embeddings, item_CF_embeddings)
        #用户嵌入，物品嵌入，用户潜在嵌入，物品初始嵌入
        return scores

    def predict(self, user_CF_embeddings, item_KG_embeddings, user_KG_embeddings, item_CF_embeddings):
        e_u = user_CF_embeddings[0]
        e_p_u = user_KG_embeddings[0]
        e_p_i = item_CF_embeddings[0]
        e_i = item_KG_embeddings[0]
        kge_loss = 0
        if len(user_CF_embeddings) != len(item_KG_embeddings):
             raise Exception("Concat aggregator needs same length for user and item embedding")
           #对比学习
        for i in range(0,  len(user_CF_embeddings)):
            kge_loss += self.MLCL_Loss(user_CF_embeddings[i],  user_KG_embeddings[i], user_CF_embeddings, user_KG_embeddings)
            kge_loss += self.MLCL_Loss(item_KG_embeddings[i], item_CF_embeddings[i], item_KG_embeddings, item_CF_embeddings)
        for i in range(1, len(user_CF_embeddings)):
            e_u = torch.cat((user_CF_embeddings[i], e_u), dim=-1)
        for i in range(1, len(user_KG_embeddings)):
            e_p_u = torch.cat((user_KG_embeddings[i], e_p_u), dim=-1)
        for i in range(1, len(item_KG_embeddings)):
            e_i = torch.cat((item_KG_embeddings[i], e_i), dim=-1)
        for i in range(1, len(item_CF_embeddings)):
            e_p_i = torch.cat((item_CF_embeddings[i], e_p_i), dim=-1)
        
        e_u = torch.cat((e_u, e_p_u), dim=-1)
        e_i = torch.cat((e_p_i, e_i), dim=-1)

        # 最终预测前应用dropout
        #e_u = self.gate_dropout(e_u)
        #e_i = self.gate_dropout(e_i)
        
        #e_u = self.gate1(e_u,e_p_u)
        #e_i= self.gate1(e_i,e_p_i)
 
        scores = (e_i * e_u).sum(dim=1)
        scores = torch.sigmoid(scores)
        return scores, kge_loss
  
    
    def _parse_args(self, args, n_entity, n_relation):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_layer = args.n_layer
        self.agg = args.agg
        self.emb_dropout_rate = args.emb_dropout
        self.att_dropout_rate = args.att_dropout
        self.gate_dropout_rate = args.gate_dropout


    def _init_weight(self):
        # init embedding
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)


        # init attention
        for layer in self.attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
            # init attention


    def ed_knowledge_agg(self, h_set_emb, path_emb, t_emb):
        att_weights = self.attention(torch.cat((h_set_emb, path_emb), dim=-1)).squeeze(-1)
        att_weights_norm = F.softmax(att_weights, dim=-1)
        emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_emb)
        emb_i = emb_i.sum(dim=1)
        return emb_i
    
    def hyper_knowledge_agg(self, h_set_emb, path_emb, t_emb):
        # 1. 输入归一化
        h_set_emb = F.normalize(h_set_emb, p=2, dim=-1)
        path_emb = F.normalize(path_emb, p=2, dim=-1)
        t_emb = F.normalize(t_emb, p=2, dim=-1)
        hyper_h_head_emb = hyper.expmap0(h_set_emb)
        hyper_h_head_emb = hyper_h_head_emb * torch.abs(self.curvature)
        hyper_tail_emb = hyper.expmap(t_emb, hyper_h_head_emb)
        hyper_relation_emb = hyper.expmap(path_emb, hyper_h_head_emb)
        tangent_input = hyper.logmap0(torch.cat((hyper_h_head_emb, hyper_relation_emb), dim=-1))     
        tangent_weights = self.attention2(tangent_input).squeeze(-1)
        att_weights_norm = F.softmax(tangent_weights, dim=-1)
        hyper_agg = torch.mul(att_weights_norm.unsqueeze(-1), hyper_tail_emb)    
        mean_point = (att_weights_norm.unsqueeze(-1) * hyper_h_head_emb).sum(dim=1,keepdim=True)
        emb_i = hyper.logmap(hyper_agg, mean_point)
        emb_i = (att_weights_norm.unsqueeze(-1) * emb_i).sum(dim=1)
        return emb_i

    # graph contrastive learning for KGE
    def MLCL_Loss(self,current_user_inital_embedding, current_user_potential_embedding, user_inital_embedding, user_KG_embeddings):
        previous_user_KG_embeddings_all = user_KG_embeddings[0]# 用户潜在嵌入的第1层
        previous_user_inital_embedding_all = user_inital_embedding[0]# 用户初始嵌入的第1层
        for i in range(1, len(user_KG_embeddings)):
            previous_user_KG_embeddings_all= torch.cat((user_KG_embeddings[i], previous_user_KG_embeddings_all), dim=0)
            #user potential embedding 聚合前i层用户潜在嵌入表示
        for i in range(1, len(user_inital_embedding)):
            previous_user_inital_embedding_all = torch.cat((user_inital_embedding[i], previous_user_inital_embedding_all), dim=0)
            #user_inital_embedding 聚合前i层用户初始嵌入表示
        #for user potential 潜在user
        #for user_inital_embedding
        #当前i层用户的嵌入表示previous_user_KG_embeddings_all
        norm_user_inital = F.normalize(current_user_inital_embedding)
        #当前i层物品的嵌入表示
        norm_user_potential = F.normalize(current_user_potential_embedding)
        #聚合前k层用户的嵌入表示
        norm_all_user_inital = F.normalize(previous_user_KG_embeddings_all)
        norm_all_user_potential = F.normalize(previous_user_inital_embedding_all)
        
        
        #内积
        pos_score_user_inital = torch.mul(norm_user_inital, norm_user_potential).sum(dim=1)
        #计算得分
        ttl_score_user_inital = torch.matmul(norm_user_inital, norm_all_user_inital.transpose(0, 1))
        #通过softmax函数计算得分
        pos_score_user_inital = torch.exp(pos_score_user_inital / self.ssl_temp)
        ttl_score_user_inital = torch.exp(ttl_score_user_inital / self.ssl_temp).sum(dim=1)
        #计算损失函数
        kge_loss_user_inital = -torch.log(pos_score_user_inital / ttl_score_user_inital).sum()
        #for item

        #for user_inital_embedding 
        pos_score_user_potential = torch.mul(norm_user_potential, norm_user_inital).sum(dim=1)
        ttl_score_user_potential = torch.matmul(norm_user_potential, norm_all_user_potential.transpose(0, 1))
        pos_score_user_potential = torch.exp(pos_score_user_potential / self.ssl_temp)
        ttl_score_user_potential = torch.exp(ttl_score_user_potential / self.ssl_temp).sum(dim=1)
        kge_loss_user_potential = -torch.log(pos_score_user_potential / ttl_score_user_potential).sum()

        kge_loss = self.kge_weight * (kge_loss_user_inital + kge_loss_user_potential)

        return kge_loss