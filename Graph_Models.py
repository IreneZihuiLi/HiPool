
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

from math import ceil
from torch_geometric.nn import dense_diff_pool
from torch_geometric.nn import DenseGCNConv


# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/sage_conv.html#SAGEConv



class GCN(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1) # shape: 2708, 7


class GAT(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=2)
        self.conv2 = GATConv(hidden_dim * 2, output_dim,heads=1)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        output = F.dropout(x, training=self.training)
        output = self.conv2(output, edge_index)

        return F.log_softmax(output, dim=1)  # shape: [num_node/x.shape[0], output_dim/num_class]

# add classic methods
class GraphSAGE(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        self.conv1 = SAGEConv(input_dim,hidden_dim)
        self.conv2 = SAGEConv(hidden_dim,output_dim)

    def forward(self,x,edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# add classic methods
class LinearFirst(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        self.conv1 = torch.nn.Linear(input_dim,output_dim)

    def forward(self,x,edge_index):

        # import pdb;pdb.set_trace()

        # keep top 10%
        topk = int(x.shape[0]*0.1)
        output = F.relu(x[0:topk])
        output = F.dropout(output, training=self.training)
        output = self.conv1(output)

        return F.log_softmax(output, dim=1)

class SimpleRank(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = torch.nn.Linear(hidden_dim,output_dim)

    def forward(self,x,edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # import pdb;
        # pdb.set_trace()

        '''keep first 10%'''
        # topk = int(x.shape[0]*0.1)
        # output = F.relu(x[0:topk])

        '''keep random 10%'''
        topk = int(x.shape[0] * 0.1)
        indices = torch.randperm(x.shape[0])[:topk]
        output = F.relu(x[indices])

        '''start calculating'''
        output = F.dropout(output, training=self.training)
        output = self.conv2(output)

        return F.log_softmax(output, dim=1)



'following two methods are diffpool: Feb, 2022'
# defines a convolution network, 3 layers each
class DiffPoolGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super(DiffPoolGNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # hidden_channels = 64
        self.convs.append(DenseGCNConv(in_channels, hidden_channels,normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(DenseGCNConv(hidden_channels, hidden_channels,normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(DenseGCNConv(hidden_channels, out_channels,normalize))
        self.bns.append(torch.nn.BatchNorm1d(out_channels))

    def forward(self, x, adj, mask=None):

        for step in range(len(self.convs)):
            res = F.relu(self.convs[step](x, adj, mask))  # shape: 32, 150, 64

            'torch.permute is not working now'
            # x = torch.permute(self.bns[step](torch.permute(res, (0, 2, 1))), (0, 2, 1))
            x = self.bns[step](res.permute((0, 2, 1))).permute((0, 2, 1))



        return x


# the main class, contains two GNNs for emb and pooling
class DiffPool(torch.nn.Module):
    def __init__(self, device,max_nodes, input_dim, hidden_dim, output_dim):
        super(DiffPool, self).__init__()

        self.device = device

        self.num_nodes1 = ceil(max_nodes*0.5)
        self.gnn1_pool = DiffPoolGNN(input_dim, hidden_dim, self.num_nodes1)
        self.gnn1_embed = DiffPoolGNN(input_dim, hidden_dim, hidden_dim)

        self.num_nodes2 = ceil(self.num_nodes1*0.5)
        self.gnn2_pool = DiffPoolGNN(hidden_dim, hidden_dim, self.num_nodes2)
        self.gnn2_embed = DiffPoolGNN(hidden_dim, hidden_dim, hidden_dim, lin=False)

        self.gnn3_embed = DiffPoolGNN(hidden_dim, hidden_dim, hidden_dim, lin=False)

        # MLP for the final layer
        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, output_dim)




    def forward(self, x, adj, mask=None):
        # add a batch dim (batch size is 1) for dense layer computation
        newadj = adj[1].unsqueeze(0).float()
        newx = x.unsqueeze(0).float()

        # s = self.gnn1_pool(newx, newadj, mask)
        x = self.gnn1_embed(newx, newadj, mask)

        'use a new method for constructing s'
        portion1 = ceil(x.shape[1]/self.num_nodes1)
        flat_s = torch.eye(self.num_nodes1)
        flat_s = torch.repeat_interleave(flat_s, portion1, dim=0)[:x.shape[1], ].unsqueeze(0).float().to(self.device)


        # x, adj, l1, e1 = dense_diff_pool(x,newadj,s,mask) # new x shape [1, num_node, hidden]; new adj shape [1,num_node, num_node]
        x, adj, l1, e1 = dense_diff_pool(x, newadj, flat_s, mask)


        # s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        # import pdb;
        # pdb.set_trace()

        portion2 = ceil(x.shape[1] / self.num_nodes2)
        flat_s = torch.eye(self.num_nodes2)
        flat_s = torch.repeat_interleave(flat_s, portion2, dim=0)[:x.shape[1], ].unsqueeze(0).float().to(self.device)


        x, adj, l2, e2 = dense_diff_pool(x, adj, flat_s)


        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)


        return F.log_softmax(x, dim=1)

from Graph_Models_utils import SpGraphAttentionLayer
'following two methods are our hi-method: Feb, 2022'
class HiPool(torch.nn.Module):
    def __init__(self,device,input_dim,hidden_dim,output_dim):
        super().__init__() # hid dim 32

        self.device = device


        self.num_nodes1 = 10
        self.num_nodes2 = ceil(self.num_nodes1/2)

        'parameterize adj: not very helpful'
        # self.adj1 = torch.nn.Parameter(torch.zeros(size=(self.num_nodes1,self.num_nodes1))).to(self.device)
        # torch.nn.init.xavier_normal_(self.adj1.data, gain=1.414)
        # self.adj2 = torch.nn.Parameter(torch.zeros(size=(self.num_nodes2, self.num_nodes2))).to(self.device)
        # torch.nn.init.xavier_normal_(self.adj2.data, gain=1.414)

        self.conv1 = DenseGCNConv(input_dim, hidden_dim)
        self.conv2 = DenseGCNConv(hidden_dim, hidden_dim)

        'GAT: not very helpful'
        # self.conv1 = SpGraphAttentionLayer(input_dim,hidden_dim)
        # self.conv2 = SpGraphAttentionLayer(hidden_dim, hidden_dim)

        # output layer
        self.linear1 = torch.nn.Linear(hidden_dim, output_dim)


        # cross-layer attention, l1
        self.cross_attention_l1 = torch.nn.Parameter(torch.zeros(size=(input_dim, input_dim))).to(self.device)
        torch.nn.init.xavier_normal_(self.cross_attention_l1.data, gain=1.414)

        # cross-layer attention, l2
        self.cross_attention_l2 = torch.nn.Parameter(torch.zeros(size=(hidden_dim, hidden_dim))).to(self.device)
        torch.nn.init.xavier_normal_(self.cross_attention_l2.data, gain=1.414)

        # reversed linear layer, l1
        self.reversed_l1 = torch.nn.Parameter(torch.zeros(size=(hidden_dim, input_dim))).to(self.device)
        torch.nn.init.xavier_normal_(self.reversed_l1.data, gain=1.414)

        self.reversed_conv1 = DenseGCNConv(input_dim, hidden_dim)

        # add self-attention for l1
        self.multihead_attn_l1 = torch.nn.MultiheadAttention(embed_dim=32, num_heads=2)


    # def forward(self, x, edge_index):
    #
    #     'Hi-pool: GCN GAT 1-2 layers, normal hierarchy, without sent-token cross-attention'
    #     newadj = edge_index[1].float()
    #
    #     portion1 = ceil(x.shape[0] / self.num_nodes1)
    #     flat_s = torch.eye(self.num_nodes1)
    #     flat_s = torch.repeat_interleave(flat_s, portion1, dim=0)[:x.shape[0], ].float().to(self.device)
    #     # (13,5)
    #
    #
    #     # first layer
    #     x1 = torch.matmul(flat_s.t(),x) # (5,128)
    #     self.adj1 = torch.matmul(torch.matmul(flat_s.t(),newadj), flat_s)
    #
    #     x = self.conv1(x1,self.adj1)
    #
    #
    #     x = F.relu(x)
    #     x = F.dropout(x, training=self.training)[0]
    #
    #     # second layer
    #     portion2 = ceil(x.shape[0] / self.num_nodes2)
    #     flat_s = torch.eye(self.num_nodes2)
    #     flat_s = torch.repeat_interleave(flat_s, portion2, dim=0)[:x.shape[0], ].float().to(self.device)
    #
    #     # import pdb;pdb.set_trace()
    #     x2 = torch.matmul(flat_s.t(), x)
    #     self.adj2 = torch.matmul(torch.matmul(flat_s.t(), self.adj1), flat_s)
    #     x = self.conv2(x2, self.adj2)
    #
    #
    #     'return mean'
    #     x = x.mean(dim=1)
    #     x = F.relu(self.linear1(x))
    #     return F.log_softmax(x, dim=1)

    def forward_best(self, x, edge_index):
        # forward_cross_best: april 7

        'hipool: add sent-token cross-attention (cross-layer) attention: 2 layers'
        newadj = edge_index[1].float()



        portion1 = ceil(x.shape[0] / self.num_nodes1)
        flat_s = torch.eye(self.num_nodes1)
        flat_s = torch.repeat_interleave(flat_s, portion1, dim=0)[:x.shape[0], ].float().to(self.device)


        # first layer
        x1 = torch.matmul(flat_s.t(),x) # (5,128)
        self.adj1 = torch.matmul(torch.matmul(flat_s.t(),newadj), flat_s)

        'testing cross-layer attention'
        # generate inverse adj for cross-layer attention
        reverse_s = torch.ones_like(flat_s) - flat_s
        scores = torch.matmul(torch.matmul(x1, self.cross_attention_l1),x.t())
        # mask own cluster and do cross-cluster
        scores = scores * reverse_s.t()
        alpha = F.softmax(scores, dim=1)
        # compute \alpha * x
        x1 = torch.matmul(alpha,x) + x1
        'cross-layer ends'


        x = self.conv1(x1,self.adj1)


        x = F.relu(x)
        x = F.dropout(x, training=self.training)[0]

        # second layer
        portion2 = ceil(x.shape[0] / self.num_nodes2)
        flat_s = torch.eye(self.num_nodes2)
        flat_s = torch.repeat_interleave(flat_s, portion2, dim=0)[:x.shape[0], ].float().to(self.device)



        x2 = torch.matmul(flat_s.t(), x)
        self.adj2 = torch.matmul(torch.matmul(flat_s.t(), self.adj1), flat_s)


        'testing cross-layer attention for 2nd layer'
        # generate inverse adj for cross-layer attention
        # reverse_s = torch.ones_like(flat_s) - flat_s
        # scores = torch.matmul(torch.matmul(x2, self.cross_attention_l2), x.t())
        # # mask own cluster and do cross-cluster
        # scores = scores * reverse_s.t()
        # alpha = F.softmax(scores, dim=1)
        # # compute \alpha * x
        # x2 = torch.matmul(alpha, x) + x2
        'cross-layer for 2nd layer ends'

        x = self.conv2(x2, self.adj2)



        'return mean'
        x = x.mean(dim=1)
        x = F.relu(self.linear1(x))
        return F.log_softmax(x, dim=1)


    # def forward(self, x, edge_index):
    #
    #     'hipool: add sent-token cross-attention (cross-layer) attention: 1 layer, the best setting so far'
    #     newadj = edge_index[1].float()
    #
    #     portion1 = ceil(x.shape[0] / self.num_nodes1)
    #     flat_s = torch.eye(self.num_nodes1)
    #     flat_s = torch.repeat_interleave(flat_s, portion1, dim=0)[:x.shape[0], ].float().to(self.device)
    #
    #
    #     # first layer
    #     x1 = torch.matmul(flat_s.t(),x) # (5,128)
    #     self.adj1 = torch.matmul(torch.matmul(flat_s.t(),newadj), flat_s)
    #
    #     'testing cross-layer attention'
    #     # generate inverse adj for cross-layer attention
    #     reverse_s = torch.ones_like(flat_s) - flat_s
    #     scores = torch.matmul(torch.matmul(x1, self.cross_attention_l1),x.t())
    #     # mask own cluster and do cross-cluster
    #     scores = scores * reverse_s.t()
    #     alpha = F.softmax(scores, dim=1)
    #     # compute \alpha * x
    #     x1 = torch.matmul(alpha,x) + x1
    #     'cross-layer ends'
    #
    #
    #     x = self.conv1(x1,self.adj1)
    #
    #
    #     x = F.relu(x)
    #     x = F.dropout(x, training=self.training)[0]
    #
    #     # second layer
    #     portion2 = ceil(x.shape[0] / self.num_nodes2)
    #     flat_s = torch.eye(self.num_nodes2)
    #     flat_s = torch.repeat_interleave(flat_s, portion2, dim=0)[:x.shape[0], ].float().to(self.device)
    #
    #     x2 = torch.matmul(flat_s.t(), x)
    #     self.adj2 = torch.matmul(torch.matmul(flat_s.t(), self.adj1), flat_s)
    #
    #     # import pdb;
    #     # pdb.set_trace()
    #
    #
    #     x = self.conv2(x2, self.adj2)
    #
    #
    #
    #     'return mean'
    #     x = x.mean(dim=1)
    #     x = F.relu(self.linear1(x))
    #     return F.log_softmax(x, dim=1)


    # def forward_reverse(self, x, edge_index):
    #
    #     'hipool: add sent-token cross-attention (cross-layer) attention: 1 layer, the best setting so far'
    #     newadj = edge_index[1].float()
    #
    #     portion1 = ceil(x.shape[0] / self.num_nodes1)
    #     flat_s = torch.eye(self.num_nodes1)
    #     flat_s = torch.repeat_interleave(flat_s, portion1, dim=0)[:x.shape[0], ].float().to(self.device)
    #
    #
    #     # first layer
    #     input_x = x
    #     x1 = torch.matmul(flat_s.t(),x) # (5,128)
    #     self.adj1 = torch.matmul(torch.matmul(flat_s.t(),newadj), flat_s)
    #
    #     'testing cross-layer attention'
    #     # generate inverse adj for cross-layer attention
    #     reverse_s = torch.ones_like(flat_s) - flat_s
    #     scores = torch.matmul(torch.matmul(x1, self.cross_attention_l1),x.t())
    #     # mask own cluster and do cross-cluster
    #     scores = scores * reverse_s.t()
    #     alpha = F.softmax(scores, dim=1)
    #     # compute \alpha * x
    #     x1 = torch.matmul(alpha,x) + x1 # x1 dim 128
    #     'cross-layer ends'
    #
    #
    #
    #
    #     x = self.conv1(x1,self.adj1)
    #
    #     x = F.relu(x)
    #     x = F.dropout(x, training=self.training)[0] # shape [5,32], mapping adj is flat_s shape [15,5]
    #
    #     # import pdb;
    #     # pdb.set_trace()
    #
    #     'map back to the previous layer x->x1'
    #     back_l1=torch.matmul(torch.matmul(flat_s,x),self.reversed_l1) # map back to input l0, same dim
    #     back_l1 = input_x + back_l1 # add original input l0, [14,128]
    #     # map with another conv layer
    #     back_l1_x = torch.matmul(flat_s.t(), back_l1)
    #     back_l1_x = self.reversed_conv1(back_l1_x,self.adj1)
    #     back_l1_x = F.relu(back_l1_x)
    #     back_l1_x = F.dropout(back_l1_x,training=self.training)[0]
    #     x = x + back_l1_x
    #     'map back ends'
    #
    #     # second layer
    #     portion2 = ceil(x.shape[0] / self.num_nodes2)
    #     flat_s = torch.eye(self.num_nodes2)
    #     flat_s = torch.repeat_interleave(flat_s, portion2, dim=0)[:x.shape[0], ].float().to(self.device)
    #
    #     x2 = torch.matmul(flat_s.t(), x)
    #     self.adj2 = torch.matmul(torch.matmul(flat_s.t(), self.adj1), flat_s)
    #
    #
    #
    #
    #     x = self.conv2(x2, self.adj2)
    #
    #     'return mean'
    #     x = x.mean(dim=1)
    #     x = F.relu(self.linear1(x))
    #     return F.log_softmax(x, dim=1)



    def forward_cross_with_self_2layer(self, x, edge_index):

        'forward_one_cross_with_self_attention: hipool: add sent-token cross-attention (cross-layer) attention: 1 layer'
        'for each higher layer, we add self attention'
        newadj = edge_index[1].float()

        portion1 = ceil(x.shape[0] / self.num_nodes1)
        flat_s = torch.eye(self.num_nodes1)
        flat_s = torch.repeat_interleave(flat_s, portion1, dim=0)[:x.shape[0], ].float().to(self.device)


        # first layer
        x1 = torch.matmul(flat_s.t(),x) # (5,128)
        self.adj1 = torch.matmul(torch.matmul(flat_s.t(),newadj), flat_s)

        'testing cross-layer attention'
        # generate inverse adj for cross-layer attention
        reverse_s = torch.ones_like(flat_s) - flat_s
        scores = torch.matmul(torch.matmul(x1, self.cross_attention_l1),x.t())
        # mask own cluster and do cross-cluster
        scores = scores * reverse_s.t()
        alpha = F.softmax(scores, dim=1)
        # compute \alpha * x
        x1 = torch.matmul(alpha,x) + x1
        'cross-layer ends'


        x = self.conv1(x1,self.adj1)


        x = F.relu(x)
        x = F.dropout(x, training=self.training)[0] # shape [self.num_nodes1, 32]

        # import pdb;
        # pdb.set_trace()
        'testing self-attention on the inner layer'
        test_x = x.unsqueeze(1)
        attn_output, _ = self.multihead_attn_l1(test_x, test_x, test_x)
        x = test_x.squeeze(1)
        'self-attention ends'

        # second layer
        portion2 = ceil(x.shape[0] / self.num_nodes2)
        flat_s = torch.eye(self.num_nodes2)
        flat_s = torch.repeat_interleave(flat_s, portion2, dim=0)[:x.shape[0], ].float().to(self.device)



        x2 = torch.matmul(flat_s.t(), x)
        self.adj2 = torch.matmul(torch.matmul(flat_s.t(), self.adj1), flat_s)


        'testing cross-layer attention for 2nd layer'
        # generate inverse adj for cross-layer attention
        reverse_s = torch.ones_like(flat_s) - flat_s
        scores = torch.matmul(torch.matmul(x2, self.cross_attention_l2), x.t())
        # mask own cluster and do cross-cluster
        scores = scores * reverse_s.t()
        alpha = F.softmax(scores, dim=1)
        # compute \alpha * x
        x2 = torch.matmul(alpha, x) + x2
        'cross-layer for 2nd layer ends'

        x = self.conv2(x2, self.adj2)



        'return mean'
        x = x.mean(dim=1)
        x = F.relu(self.linear1(x))
        return F.log_softmax(x, dim=1)

    def forward(self, x, edge_index):
        # forward_cross_best

        'hipool: add sent-token cross-attention (cross-layer) attention: 2 layers'
        newadj = edge_index[1].float()
        portion1 = ceil(x.shape[0] / self.num_nodes1)
        flat_s = torch.eye(self.num_nodes1)
        flat_s = torch.repeat_interleave(flat_s, portion1, dim=0)[:x.shape[0], ].float().to(self.device)

        # first layer
        x1 = torch.matmul(flat_s.t(), x)  # (5,128)
        self.adj1 = torch.matmul(torch.matmul(flat_s.t(), newadj), flat_s)

        'testing cross-layer attention'
        # generate inverse adj for cross-layer attention
        reverse_s = torch.ones_like(flat_s) - flat_s
        scores = torch.matmul(torch.matmul(x1, self.cross_attention_l1), x.t())
        # mask own cluster and do cross-cluster
        scores = scores * reverse_s.t()
        alpha = F.softmax(scores, dim=1)
        # compute \alpha * x
        x1 = torch.matmul(alpha, x) + x1
        'cross-layer ends'

        x = self.conv1(x1, self.adj1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)[0]

        # second layer
        portion2 = ceil(x.shape[0] / self.num_nodes2)
        flat_s = torch.eye(self.num_nodes2)
        flat_s = torch.repeat_interleave(flat_s, portion2, dim=0)[:x.shape[0], ].float().to(self.device)

        x2 = torch.matmul(flat_s.t(), x)
        self.adj2 = torch.matmul(torch.matmul(flat_s.t(), self.adj1), flat_s)

        'testing cross-layer attention for 2nd layer'
        # generate inverse adj for cross-layer attention
        reverse_s = torch.ones_like(flat_s) - flat_s
        scores = torch.matmul(torch.matmul(x2, self.cross_attention_l2), x.t())
        # mask own cluster and do cross-cluster
        scores = scores * reverse_s.t()
        alpha = F.softmax(scores, dim=1)
        # compute \alpha * x
        x2 = torch.matmul(alpha, x) + x2
        'cross-layer for 2nd layer ends'

        x = self.conv2(x2, self.adj2)

        'return mean'
        x = x.mean(dim=1)
        x = F.relu(self.linear1(x))



        return F.log_softmax(x, dim=1)

