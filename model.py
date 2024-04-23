import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class GAT_GCN(torch.nn.Module):
    def __init__(self, graph_output_dim=128, graph_features_dim=78, dropout=0.2):
        super(GAT_GCN, self).__init__()
        self.conv1 = GATConv(graph_features_dim, graph_features_dim, heads=10)
        self.conv2 = GCNConv(graph_features_dim * 10, graph_features_dim * 10)
        self.fc_g1 = torch.nn.Linear(graph_features_dim * 10 * 2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, graph_output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        return x


class GATNet(torch.nn.Module):
    def __init__(self, num_features_xd=78, n_output=1, num_features_xt=25,
                     n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(GATNet, self).__init__()

        # graph layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc_xt1 = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)          # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)

        # protein input feed-forward:
        target = data.target
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt1(embedded_xt)
        conv_xt = self.relu(conv_xt)

        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc_xt1(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


class SeqNet(torch.nn.Module):
    def __init__(self, seq_embed_dim=1024, n_filters=256, seq_output_dim=1024, dropout=0.2):
        super(SeqNet, self).__init__()

        self.conv_xt_1 = nn.Conv1d(in_channels=seq_embed_dim, out_channels=n_filters, kernel_size=5)
        self.pool_xt_1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=seq_embed_dim, kernel_size=5)
        self.pool_xt_2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_xt_3 = nn.Conv1d(in_channels=seq_embed_dim, out_channels=n_filters, kernel_size=5)
        self.pool_xt_3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_xt_4 = nn.Conv1d(in_channels=n_filters, out_channels=int(n_filters / 2), kernel_size=3)
        self.pool_xt_4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1_xt = nn.Linear(128 * 61, seq_output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_embed, seq_mask=None):
        # 1d conv layers
        xt = self.conv_xt_1(seq_embed.transpose(1, 2))
        xt = self.relu(xt)
        xt = self.pool_xt_1(xt)
        xt = self.conv_xt_2(xt)
        xt = self.relu(xt)
        xt = self.pool_xt_2(xt)
        xt = self.conv_xt_3(xt)
        xt = self.relu(xt)
        xt = self.pool_xt_3(xt)
        xt = self.conv_xt_4(xt)
        xt = self.relu(xt)
        xt = self.pool_xt_4(xt)

        # flatten
        xt = xt.view(-1, 128 * 61)
        xt = self.fc1_xt(xt)
        xt = self.relu(xt)
        xt = self.dropout(xt)
        return xt


class DGSDTAModel(torch.nn.Module):
    def __init__(self, config):
        super(DGSDTAModel, self).__init__()
        dropout = config['dropout']
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # SMILES graph branch
        if config['graphNet'] == 'GAT_GCN':
            self.graph = GAT_GCN(config['graph_output_dim'], config['graph_features_dim'], dropout)
        elif config['graphNet'] == 'GAT':
            self.graph = GATNet(config['graph_output_dim'], config['graph_features_dim'], dropout)
        else:
            print("Unknow model name")

        # Seq branch
        self.seqnet = SeqNet(config['seq_embed_dim'], config['n_filters'], config['seq_output_dim'], dropout)

        # combined layers
        self.fc1 = nn.Linear(config['graph_output_dim'] + config['seq_output_dim'], 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, 1)
        # self.bilstm = nn.LSTM(64, 64, num_layers=1, bidirectional=True, dropout=self.dropout)

    def forward(self, graph, seq_embed, seq_mask=None):
        graph_output = self.graph(graph)
        seq_output = self.seqnet(seq_embed, seq_mask)
        # concat
        # h_0 = Variable(torch.zeros(2, 1, 64).to(device))
        # c_0 = Variable(torch.zeros(2, 1, 64).to(device))
        # #
        # seq_output, _ = self.bilstm(seq_output, (h_0, c_0))
        # seq_output =
        xc = torch.cat((graph_output, seq_output), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        out = self.out(xc)
        return out


if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    import pickle
    from dataset import DGSDTADataset
    from config import DGSDTAModelConfig

    with open('data/seq2path_prot_albert.pickle', 'rb') as handle:
        seq2path = pickle.load(handle)
    with open('data/smile_graph.pickle', 'rb') as f:
        smile2graph = pickle.load(f)

    train_dataset = DGSDTADataset('data/davis_train.csv', smile2graph, seq2path)
    loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)

    device = torch.device("cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    config = DGSDTAModelConfig()
    print(config['graphNet'])
    model = DGSDTAModel(config).to(device)

    for i, data in enumerate(loader):
        graph, seq_embed, seq_mask = data
        graph = graph.to(device)
        seq_embed = seq_embed.to(device)
        seq_mask = seq_mask.to(device)
        out = model(graph, seq_embed)
        # out = graph_(graph)
        print(out.shape)
        break



