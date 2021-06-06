import os
import time
import glob
import argparse
import os.path as osp
from tqdm import tqdm

from typing import Optional, List, NamedTuple

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Dropout
from torch.optim.lr_scheduler import StepLR

from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)

from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import NeighborSampler

from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from root import ROOT


class Batch(NamedTuple):
    x: Tensor
    y: Tensor
    adjs_t: List[SparseTensor]

    def to(self, *args, **kwargs):
        return Batch(
            x=self.x.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            adjs_t=[adj_t.to(*args, **kwargs) for adj_t in self.adjs_t],
        )


def get_col_slice(x, start_row_idx, end_row_idx, start_col_idx, end_col_idx):
    outs = []
    chunk = 100000
    for i in tqdm(range(start_row_idx, end_row_idx, chunk)):
        j = min(i + chunk, end_row_idx)
        outs.append(x[i:j, start_col_idx:end_col_idx].copy())
    return np.concatenate(outs, axis=0)


def save_col_slice(x_src, x_dst, start_row_idx, end_row_idx, start_col_idx,
                   end_col_idx):
    assert x_src.shape[0] == end_row_idx - start_row_idx
    assert x_src.shape[1] == end_col_idx - start_col_idx
    chunk, offset = 100000, start_row_idx
    for i in tqdm(range(0, end_row_idx - start_row_idx, chunk)):
        j = min(i + chunk, end_row_idx - start_row_idx)
        x_dst[offset + i:offset + j, start_col_idx:end_col_idx] = x_src[i:j]


class MAG240M(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, sizes: List[int],
                 in_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sizes = sizes
        self.in_memory = in_memory

    @property
    def num_features(self) -> int:
        return 768

    @property
    def num_classes(self) -> int:
        return 153

    @property
    def num_relations(self) -> int:
        return 5

    def prepare_data(self):
        dataset = MAG240MDataset(self.data_dir)

        path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
        if not osp.exists(path):  # Will take approximately 5 minutes...
            t = time.perf_counter()
            print('Converting adjacency matrix...', end=' ', flush=True)
            edge_index = dataset.edge_index('paper', 'cites', 'paper')
            edge_index = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=edge_index[0], col=edge_index[1],
                sparse_sizes=(dataset.num_papers, dataset.num_papers),
                is_sorted=True)
            torch.save(adj_t.to_symmetric(), path)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        path = f'{dataset.dir}/full_adj_t.pt'
        if not osp.exists(path):  # Will take approximately 16 minutes...
            t = time.perf_counter()
            print('Merging adjacency matrices...', end=' ', flush=True)

            row, col, _ = torch.load(
                f'{dataset.dir}/paper_to_paper_symmetric.pt').coo()
            rows, cols = [row], [col]

            edge_index = dataset.edge_index('author', 'writes', 'paper')
            row, col = torch.from_numpy(edge_index)
            row += dataset.num_papers
            rows += [row, col]
            cols += [col, row]

            edge_index = dataset.edge_index('author', 'institution')
            row, col = torch.from_numpy(edge_index)
            row += dataset.num_papers
            col += dataset.num_papers + dataset.num_authors
            rows += [row, col]
            cols += [col, row]

            edge_types = [
                torch.full(x.size(), i, dtype=torch.int8)
                for i, x in enumerate(rows)
            ]

            row = torch.cat(rows, dim=0)
            del rows
            col = torch.cat(cols, dim=0)
            del cols

            N = (dataset.num_papers + dataset.num_authors +
                 dataset.num_institutions)

            perm = (N * row).add_(col).numpy().argsort()
            perm = torch.from_numpy(perm)
            row = row[perm]
            col = col[perm]

            edge_type = torch.cat(edge_types, dim=0)[perm]
            del edge_types

            full_adj_t = SparseTensor(row=row, col=col, value=edge_type,
                                      sparse_sizes=(N, N), is_sorted=True)

            torch.save(full_adj_t, path)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        path = f'{dataset.dir}/full_feat.npy'
        done_flag_path = f'{dataset.dir}/full_feat_done.txt'
        if not osp.exists(done_flag_path):  # Will take ~3 hours...
            t = time.perf_counter()
            print('Generating full feature matrix...')

            node_chunk_size = 100000
            dim_chunk_size = 64
            N = (dataset.num_papers + dataset.num_authors +
                 dataset.num_institutions)

            paper_feat = dataset.paper_feat
            x = np.memmap(path, dtype=np.float16, mode='w+',
                          shape=(N, self.num_features))

            print('Copying paper features...')
            for i in tqdm(range(0, dataset.num_papers, node_chunk_size)):
                j = min(i + node_chunk_size, dataset.num_papers)
                x[i:j] = paper_feat[i:j]

            edge_index = dataset.edge_index('author', 'writes', 'paper')
            row, col = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=row, col=col,
                sparse_sizes=(dataset.num_authors, dataset.num_papers),
                is_sorted=True)

            # Processing 64-dim subfeatures at a time for memory efficiency.
            print('Generating author features...')
            for i in tqdm(range(0, self.num_features, dim_chunk_size)):
                j = min(i + dim_chunk_size, self.num_features)
                inputs = get_col_slice(paper_feat, start_row_idx=0,
                                       end_row_idx=dataset.num_papers,
                                       start_col_idx=i, end_col_idx=j)
                inputs = torch.from_numpy(inputs)
                outputs = adj_t.matmul(inputs, reduce='mean').numpy()
                del inputs
                save_col_slice(
                    x_src=outputs, x_dst=x, start_row_idx=dataset.num_papers,
                    end_row_idx=dataset.num_papers + dataset.num_authors,
                    start_col_idx=i, end_col_idx=j)
                del outputs

            edge_index = dataset.edge_index('author', 'institution')
            row, col = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=col, col=row,
                sparse_sizes=(dataset.num_institutions, dataset.num_authors),
                is_sorted=False)

            print('Generating institution features...')
            # Processing 64-dim subfeatures at a time for memory efficiency.
            for i in tqdm(range(0, self.num_features, dim_chunk_size)):
                j = min(i + dim_chunk_size, self.num_features)
                inputs = get_col_slice(
                    x, start_row_idx=dataset.num_papers,
                    end_row_idx=dataset.num_papers + dataset.num_authors,
                    start_col_idx=i, end_col_idx=j)
                inputs = torch.from_numpy(inputs)
                outputs = adj_t.matmul(inputs, reduce='mean').numpy()
                del inputs
                save_col_slice(
                    x_src=outputs, x_dst=x,
                    start_row_idx=dataset.num_papers + dataset.num_authors,
                    end_row_idx=N, start_col_idx=i, end_col_idx=j)
                del outputs

            x.flush()
            del x
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

            with open(done_flag_path, 'w') as f:
                f.write('done')

    def setup(self, stage: Optional[str] = None):
        t = time.perf_counter()                                                 #time it takes to perform
        print('Reading dataset...', end=' ', flush=True)
        dataset = MAG240MDataset(self.data_dir)                                 #read data directory into dataset

        self.train_idx = torch.from_numpy(dataset.get_idx_split('train'))       #get split
        self.train_idx = self.train_idx         #train_idk is a list of indices for X, see sklearn fit(X,Y)
        self.train_idx.share_memory_()          #moves storage to memory, CUDA
        self.val_idx = torch.from_numpy(dataset.get_idx_split('valid'))         #get values of indices for X as a Tensor 
        self.val_idx.share_memory_()            #moves storage to memory
        self.test_idx = torch.from_numpy(dataset.get_idx_split('test'))         #get list of indices for X to build test data as a Tensor
        self.test_idx.share_memory_()           #moves storage to memory

        N = dataset.num_papers + dataset.num_authors + dataset.num_institutions  #number of total papers, authors, institutions in the dataset       #

        x = np.memmap(f'{dataset.dir}/full_feat.npy', dtype=np.float16,         #create a read-only memory map from the file 'full_feat.npy'
                      mode='r', shape=(N, self.num_features))                   #memmap is similar to an array

        if self.in_memory:                                                      #if MAG240M is stored in memory
            self.x = np.empty((N, self.num_features), dtype=np.float16)         #create an empty array of size of features in MAG240M
            self.x[:] = x                       #fill the empty array with the memmap x
            self.x = torch.from_numpy(self.x).share_memory_()                   #turn the array x into a Tensor then move it to memory
        else:                                   #if MAG240M is not in memory
            self.x = x                          #assign the memmap x to MAG240M.x

        self.y = torch.from_numpy(dataset.all_paper_label)                      #create a Tensor of the labels of all papers and assign to MAG240M.y

        path = f'{dataset.dir}/full_adj_t.pt'                                   #create a String path to the file full_adj_t.pt
        self.adj_t = torch.load(path)           #unpickle the file full_adj_t.pt and assign to MAG240.adj_t
        print(f'Done! [{time.perf_counter() - t:.2f}s]')                        #print operation time of setup function

    def train_dataloader(self):                 #test best validation model?
        return NeighborSampler(self.adj_t, node_idx=self.train_idx,             #return train_idx adj_t node embeddings based on neighborhood
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, shuffle=True,
                               num_workers=4)

    def val_dataloader(self):                   #test best validation model?
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,               #return val_idx adj_t node embeddings based on neighborhood
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):  # Test best validation model once again.
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,               #return val_idx adj_t node embeddings based on neighborhood
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=2)

    def hidden_test_dataloader(self):           #same as test_dataloader(self) but num_workers = 3
        return NeighborSampler(self.adj_t, node_idx=self.test_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=3)

    def convert_batch(self, batch_size, n_id, adjs):
        if self.in_memory:                      #if MAG240M is in memory
            x = self.x[n_id].to(torch.float)    #convert Tensor at x[n_id] to float
        else:                                   #if MAG240M is not in memory
            x = torch.from_numpy(self.x[n_id.numpy()]).to(torch.float)          #convert data at x[n_id] to an array then to a Tensor then to a float
        y = self.y[n_id[:batch_size]].to(torch.long)                            #convert Tensor at y[n_id] from [0 to batch_size] to long
        return Batch(x=x, y=y, adjs_t=[adj_t for adj_t, _, _ in adjs])          #batch x, y, and the list adj_t


class RGNN(LightningModule):
    def __init__(self, model: str, in_channels: int, out_channels: int,         #initialize RGNN
                 hidden_channels: int, num_relations: int, num_layers: int,
                 heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = model.lower()
        self.num_relations = num_relations
        self.dropout = dropout

        self.convs = ModuleList()               #initialize convs as a module state
        self.norms = ModuleList()               #initialize norms as a module state
        self.skips = ModuleList()               #initialize skips as a module state

        if self.model == 'rgat':                #if model is RGAT
            self.convs.append(                  #add GATConv module state to convs
                ModuleList([
                    GATConv(in_channels, hidden_channels // heads, heads,       #hidden channels are heads?
                            add_self_loops=False) for _ in range(num_relations)
                ]))

            for _ in range(num_layers - 1):     #for RGNN in range num_layers-1
                self.convs.append(              #add GATConv module states to convs
                    ModuleList([
                        GATConv(hidden_channels, hidden_channels // heads,      #hidden channels are heads?
                                heads, add_self_loops=False)
                        for _ in range(num_relations)
                    ]))

        elif self.model == 'rgraphsage':        #else if model is RGraphSAGE
            self.convs.append(                  #add SAGEConv module state to convs
                ModuleList([
                    SAGEConv(in_channels, hidden_channels, root_weight=False)
                    for _ in range(num_relations)
                ]))

            for _ in range(num_layers - 1):     #for RGNN in range of num_layers-1
                self.convs.append(              #add SAGEConv module states to convs
                    ModuleList([
                        SAGEConv(hidden_channels, hidden_channels,
                                 root_weight=False)
                        for _ in range(num_relations)
                    ]))

        for _ in range(num_layers):             #for RGNN in range of num_layers
            self.norms.append(BatchNorm1d(hidden_channels))         #initialize 1D Batch norms module of hidden channels and add to norms

        self.skips.append(Linear(in_channels, hidden_channels))     #initialize Linear module of hidden channels and add to skips
        for _ in range(num_layers - 1):         #for RGNN in range of num_layers-1
            self.skips.append(Linear(hidden_channels, hidden_channels))         #initialize Linear module of hidden channels and add to skips

        self.mlp = Sequential(                  #initialize mlp as a Sequential containing the Linear, Batch norms of hidden channels
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(inplace=True),
            Dropout(p=self.dropout),
            Linear(hidden_channels, out_channels),
        )

        self.train_acc = Accuracy()             #initialize train_acc as Accuracy
        self.val_acc = Accuracy()               #initialize val_acc as Accuracy
        self.test_acc = Accuracy()              #initialize test_acc as Accuracy

    def forward(self, x: Tensor, adjs_t: List[SparseTensor]) -> Tensor:
        for i, adj_t in enumerate(adjs_t):      #for i in adjs_t
            x_target = x[:adj_t.size(0)]        #get Tensor from x[0-adj_t.size]

            out = self.skips[i](x_target)       #out = skips at i of Tensor x_target
            for j in range(self.num_relations):         #for range num_relations
                edge_type = adj_t.storage.value() == j          #edge_type = adj_t.storage value = j
                subadj_t = adj_t.masked_select_nnz(edge_type, layout='coo')         #subadj_t is Tensor that indexes the masked edge_type Tensor
                if subadj_t.nnz() > 0:          #if nonzero entries of subadj_t is greater than 0
                    out += self.convs[i][j]((x, x_target), subadj_t)                #get module at [i][j] in convs and add to out

            x = self.norms[i](out)              #x is module at norms[i]
            x = F.elu(x) if self.model == 'rgat' else F.relu(x)         #if model is RGAT, apply elu to x, else apply relu to x
            x = F.dropout(x, p=self.dropout, training=self.training)            #randomly apply zeros to elements of x during training

        return self.mlp(x)          #return the module of Sequential mlp at Tensor x

    def training_step(self, batch, batch_idx: int):        
        y_hat = self(batch.x, batch.adjs_t)                     #y_hat is RGNN of batch of x and batch of adjs_t
        train_loss = F.cross_entropy(y_hat, batch.y)            #compute cross entropy loss of y_hat and batch.y
        self.train_acc(y_hat.softmax(dim=-1), batch.y)          #get training accuracy of softmax of y_hat and batch of y
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,         #log training accuracy
                 on_epoch=True) 
        return train_loss           #return cross-entropy loss of training data

    def validation_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)                     #y_hat is RGNN of batch of x and batch of adjs_t
        self.val_acc(y_hat.softmax(dim=-1), batch.y)            #get validation accuracy of softmax of y_hat and batch of y
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True,             #log validation accuracy
                 prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx: int):                 
        y_hat = self(batch.x, batch.adjs_t)                     #y_hat is RGNN of batch of x and batch of adjs_t
        self.test_acc(y_hat.softmax(dim=-1), batch.y)           #get testing accuracy of softmax of y_hat and batch of y
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True,           #Log testing accuracy
                 prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)           #create new optimizer
        scheduler = StepLR(optimizer, step_size=25, gamma=0.25)             #create new StepLR with optimizer
        return [optimizer], [scheduler]


if __name__ == '__main__':          #if name of RGNN is '__main__'
    parser = argparse.ArgumentParser()          #create new argument parser
    parser.add_argument('--hidden_channels', type=int, default=1024)            #add argument for hidden channels
    parser.add_argument('--batch_size', type=int, default=1024)         #add argument for batch size
    parser.add_argument('--dropout', type=float, default=0.5)           #add argument for dropout
    parser.add_argument('--epochs', type=int, default=100)              #add argument for epochs
    parser.add_argument('--model', type=str, default='rgat',            #add argument for model type
                        choices=['rgat', 'rgraphsage'])
    parser.add_argument('--sizes', type=str, default='25-15')           #add argument for sizes
    parser.add_argument('--in-memory', action='store_true')             #add argument for in-memory state
    parser.add_argument('--device', type=str, default='0')              #add argument of device type as String
    parser.add_argument('--evaluate', action='store_true')              #add argument for evaluate state
    args = parser.parse_args()          #parse arguments
    args.sizes = [int(i) for i in args.sizes.split('-')]            #get size of arguments
    print(args)         #print arguments

    seed_everything(42)
    datamodule = MAG240M(ROOT, args.batch_size, args.sizes, args.in_memory)

    if not args.evaluate:
        model = RGNN(args.model, datamodule.num_features,
                     datamodule.num_classes, args.hidden_channels,
                     datamodule.num_relations, num_layers=len(args.sizes),
                     dropout=args.dropout)
        print(f'#Params {sum([p.numel() for p in model.parameters()])}')
        checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode = 'max', save_top_k=1)
        trainer = Trainer(gpus=args.device, max_epochs=args.epochs,
                          callbacks=[checkpoint_callback],
                          default_root_dir=f'logs/{args.model}')
        trainer.fit(model, datamodule=datamodule)

    if args.evaluate:
        dirs = glob.glob(f'logs/{args.model}/lightning_logs/*')
        version = max([int(x.split(os.sep)[-1].split('_')[-1]) for x in dirs])
        logdir = f'logs/{args.model}/lightning_logs/version_{version}'
        print(f'Evaluating saved model in {logdir}...')
        ckpt = glob.glob(f'{logdir}/checkpoints/*')[0]

        trainer = Trainer(gpus=args.device, resume_from_checkpoint=ckpt)
        model = RGNN.load_from_checkpoint(
            checkpoint_path=ckpt, hparams_file=f'{logdir}/hparams.yaml')

        datamodule.batch_size = 16
        datamodule.sizes = [160] * len(args.sizes)  # (Almost) no sampling...

        trainer.test(model=model, datamodule=datamodule)

        evaluator = MAG240MEvaluator()
        loader = datamodule.hidden_test_dataloader()

        model.eval()
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        y_preds = []
        for batch in tqdm(loader):
            batch = batch.to(device)
            with torch.no_grad():
                out = model(batch.x, batch.adjs_t).argmax(dim=-1).cpu()
                y_preds.append(out)
        res = {'y_pred': torch.cat(y_preds, dim=0)}
        evaluator.save_test_submission(res, f'results/{args.model}')
