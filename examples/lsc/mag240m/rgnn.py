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
    outs = []           #empty list
    chunk = 100000      #100K chunks
    for i in tqdm(range(start_row_idx, end_row_idx, chunk)):    #load up the range from start_row_indx, to end_row_index, skip by chunk
        j = min(i + chunk, end_row_idx)         #return smallest element from i + chunk
        outs.append(x[i:j, start_col_idx:end_col_idx].copy())   #add onto the back, [i:j] tp [start_col_indx to end_col_idx] shallowCopy
    return np.concatenate(outs, axis=0)         #Join list out, on axis 0


def save_col_slice(x_src, x_dst, start_row_idx, end_row_idx, start_col_idx, end_col_idx):
    assert x_src.shape[0] == end_row_idx - start_row_idx        #assert = T/F. If true run it.
    #first demension of list (x,) = end_row_indx- - start_row_idx
    assert x_src.shape[1] == end_col_idx - start_col_idx  #second demension of list (,y) = end_row_indx- - start_row_idx
    chunk, offset = 100000, start_row_idx                 #chunk = 100K, offset = start_row_index
    for i in tqdm(range(0, end_row_idx - start_row_idx, chunk)):    #load up from 0 to end_row_indx - start_row_indx, skip by chunk
        j = min(i + chunk, end_row_idx - start_row_idx)             #return smallest element from i+chunk, to end_row_index - start_row_index
        x_dst[offset + i:offset + j, start_col_idx:end_col_idx] = x_src[i:j]
        #x_dst starting from offset +  [i:offset] + j to [start_col_index:end_col_index] = x_src[i:j]

class MAG240M(LightningDataModule): #Lighting - A DataModule standardizes the training, val, test splits, data preparation and transforms.#
    #The main advantage is consistent data splits, data preparation and transforms across models.
    def __init__(self, data_dir: str, batch_size: int, sizes: List[int],   in_memory: bool = False):
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

        path = f'{dataset.dir}/paper_to_paper_symmetric.pt'     #path = dataset directory/paper_to_paper_symmetric.pt
        if not osp.exists(path):  # Will take approximately 5 minutes...
            t = time.perf_counter()     #t = float value time
            print('Converting adjacency matrix...', end=' ', flush=True)
            edge_index = dataset.edge_index('paper', 'cites', 'paper')      #datasets edges are 'paper','citations','paper'
            edge_index = torch.from_numpy(edge_index)                       #edge_index = created tensor of edge_indexes
            adj_t = SparseTensor(                                           #adjacent tensor? = tensor with a majority 0s
                row=edge_index[0], col=edge_index[1],
                sparse_sizes=(dataset.num_papers, dataset.num_papers),
                is_sorted=True)
            torch.save(adj_t.to_symmetric, path)                          #save adj_t. idk, in the path
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        path = f'{dataset.dir}/full_adj_t.pt'                             #path = data set directory/full_adj_t.pt
        if not osp.exists(path):  # Will take approximately 16 minutes...   #check if path exists
            t = time.perf_counter()     #t = float timer
            print('Merging adjacency matrices...', end=' ', flush=True)

            row, col, _ = torch.load(f'{dataset.dir}/paper_to_paper_symmetric.pt').coo()    #row and col = loaded dataset directory. COOridinate(ijv) or (tripley)
            rows, cols = [row], [col]   #turn the rows / columns into lists

            edge_index = dataset.edge_index('author', 'writes', 'paper')    #edges = datasets edges (author, writes, paper)
            row, col = torch.from_numpy(edge_index) #row, col = created tensor from edges
            row += dataset.num_papers   #row += datasets number of papers
            rows += [row, col]          #rows += listed [row,col]
            cols += [col, row]          #col += lists [col,row]

            edge_index = dataset.edge_index('author', 'institution')    #change edge index to (author, institution)
            row, col = torch.from_numpy(edge_index)                     #row, col = created densor from edge_index
            row += dataset.num_papers                                   #row += datasets number of papers
            col += dataset.num_papers + dataset.num_authors             #col += datasets number of papers + datasets number of authors
            rows += [row, col]                                          #rows += lists [row,col]
            cols += [col, row]                                          #cols += lists [col,row]

            edge_types = [torch.full(x.size(), i, dtype=torch.int8)     #create a tensor filled with x's size, 8 bit integer signed
                for i, x in enumerate(rows)                             #enumerate keeps track of the index, and the value at that index
            ]

            row = torch.cat(rows, dim=0)                                #row = cancatenated tensor rows, dimension in which the tensors are concatenated = 0
            del rows                                                    #delete rows (null)
            col = torch.cat(cols, dim=0)                                #col = cancatenated tensor 
            del cols                                                    #delete cols (null)

            N = (dataset.num_papers + dataset.num_authors + dataset.num_institutions)   #N = datasets number of papers + datasets num of authors + datasets number of insitutions

            perm = (N * row).add_(col).numpy().argsort()          # perm = (N * row) + (col). takes tensor and turns into numpy array and sorty
            perm = torch.from_numpy(perm)                         # perm = creates tesnor from perm
            row = row[perm]                                       # row = list row[perm]
            col = col[perm]                                       # col = list col[perm]

            edge_type = torch.cat(edge_types, dim=0)[perm]        # edge_type = concatenated tensor edge_types at dimensions = 0. idk what [perm]
            del edge_types                                        # delete edge_types (null)

            full_adj_t = SparseTensor(row=row, col=col, value=edge_type,  sparse_sizes=(N, N), is_sorted=True)

            torch.save(full_adj_t, path)                              #saves full_adj_t to disk file path
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        path = f'{dataset.dir}/full_feat.npy'                           #path = dataset directory/full_feat.npy                  
        done_flag_path = f'{dataset.dir}/full_feat_done.txt'            #done_flag_path = dataset directory/full_feat_done.txt
        if not osp.exists(done_flag_path):  # Will take ~3 hours...     #if file directory exists
            t = time.perf_counter()                                     #t = float time seconds
            print('Generating full feature matrix...')

            node_chunk_size = 100000
            dim_chunk_size = 64
            N = (dataset.num_papers + dataset.num_authors +  dataset.num_institutions)

            paper_feat = dataset.paper_feat                             #paper_features = datasets paper_features
            x = np.memmap(path, dtype=np.float16, mode='w+', shape=(N, self.num_features))  #x = memory map to an array stored in a binary file on disk

            print('Copying paper features...')
            for i in tqdm(range(0, dataset.num_papers, node_chunk_size)):   #load up form index -, to data.num_papers, skip by node_chunk_size
                j = min(i + node_chunk_size, dataset.num_papers)            #return smallest item from (i+node+chunk to datasets number of papers)
                x[i:j] = paper_feat[i:j]                                    #x read from [i:j] = paper_feat read from [i:j]

            edge_index = dataset.edge_index('author', 'writes', 'paper')    #edge_index = (author, writes, paper)
            row, col = torch.from_numpy(edge_index)                         #row,col = created tensor from edge_index
            adj_t = SparseTensor(  row=row, col=col, sparse_sizes=(dataset.num_authors, dataset.num_papers), is_sorted=True)

            # Processing 64-dim subfeatures at a time for memory efficiency.
            print('Generating author features...')
            for i in tqdm(range(0, self.num_features, dim_chunk_size)):     #load up from index 0, to self.num_features, skip by chunk_size
                j = min(i + dim_chunk_size, self.num_features)              #return smallest item from (i+dim_size to self.num_features)
                inputs = get_col_slice(paper_feat, start_row_idx=0,
                                       end_row_idx=dataset.num_papers,
                                       start_col_idx=i, end_col_idx=j)
                inputs = torch.from_numpy(inputs)                           #create a tensor from inputs
                outputs = adj_t.matmul(inputs, reduce='mean').numpy()       #outputs = adjt_t (line 164), return matrix product of 2 arrays, reduce = 'mean' idk, turn it into numpy array object
                del inputs                                                  #delete inputs (null)
                save_col_slice(
                    x_src=outputs, x_dst=x, start_row_idx=dataset.num_papers,
                    end_row_idx=dataset.num_papers + dataset.num_authors,
                    start_col_idx=i, end_col_idx=j)
                del outputs

            edge_index = dataset.edge_index('author', 'institution')        #edge_index = (author, institutions)
            row, col = torch.from_numpy(edge_index)
            adj_t = SparseTensor(Row=col, col=row,sparse_sizes=(dataset.num_institutions, dataset.num_authors), is_sorted=False)

            print('Generating institution features...')
            # Processing 64-dim subfeatures at a time for memory efficiency.
            for i in tqdm(range(0, self.num_features, dim_chunk_size)):     #load up from index 0, to num_features, skip dimension_chunk_size
                j = min(i + dim_chunk_size, self.num_features)              #reurn min value of (i+dim_chunk_size to num_features)
                inputs = get_col_slice(
                    x, start_row_idx=dataset.num_papers,
                    end_row_idx=dataset.num_papers + dataset.num_authors,
                    start_col_idx=i, end_col_idx=j) 
                inputs = torch.from_numpy(inputs)                           #turn inputs into a numpy array 
                outputs = adj_t.matmul(inputs, reduce='mean').numpy()       #outputs = adj_t (line 164), return matreix product of 2 arrays adn turn into a numpy array
                del inputs                                                  #delete inputs 
                save_col_slice(
                    x_src=outputs, x_dst=x,
                    start_row_idx=dataset.num_papers + dataset.num_authors,
                    end_row_idx=N, start_col_idx=i, end_col_idx=j)
                del outputs

            x.flush()                                                       #flushes internal buffer
            del x                                                           #delete x
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

            with open(done_flag_path, 'w') as f:
                f.write('done')
   
    def setup(self, stage: Optional[str] = None):
        t = time.perf_counter()
        print('Reading dataset...', end=' ', flush=True)
        dataset = MAG240MDataset(self.data_dir)

        self.train_idx = torch.from_numpy(dataset.get_idx_split('train'))
        self.train_idx = self.train_idx
        self.train_idx.share_memory_()
        self.val_idx = torch.from_numpy(dataset.get_idx_split('valid'))
        self.val_idx.share_memory_()
        self.test_idx = torch.from_numpy(dataset.get_idx_split('test'))
        self.test_idx.share_memory_()

        N = dataset.num_papers + dataset.num_authors + dataset.num_institutions

        x = np.memmap(f'{dataset.dir}/full_feat.npy', dtype=np.float16,
                      mode='r', shape=(N, self.num_features))

        if self.in_memory:
            self.x = np.empty((N, self.num_features), dtype=np.float16)
            self.x[:] = x
            self.x = torch.from_numpy(self.x).share_memory_()
        else:
            self.x = x

        self.y = torch.from_numpy(dataset.all_paper_label)

        path = f'{dataset.dir}/full_adj_t.pt'
        self.adj_t = torch.load(path)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    def train_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.train_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, shuffle=True,
                               num_workers=4)

    def val_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):  # Test best validation model once again.
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=2)

    def hidden_test_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.test_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=3)

    def convert_batch(self, batch_size, n_id, adjs):
        if self.in_memory:
            x = self.x[n_id].to(torch.float)
        else:
            x = torch.from_numpy(self.x[n_id.numpy()]).to(torch.float)
        y = self.y[n_id[:batch_size]].to(torch.long)
        return Batch(x=x, y=y, adjs_t=[adj_t for adj_t, _, _ in adjs])


class RGNN(LightningModule):
    def __init__(self, model: str, in_channels: int, out_channels: int,
                 hidden_channels: int, num_relations: int, num_layers: int,
                 heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = model.lower()
        self.num_relations = num_relations
        self.dropout = dropout

        self.convs = ModuleList()
        self.norms = ModuleList()
        self.skips = ModuleList()

        if self.model == 'rgat':
            self.convs.append(
                ModuleList([
                    GATConv(in_channels, hidden_channels // heads, heads,
                            add_self_loops=False) for _ in range(num_relations)
                ]))

            for _ in range(num_layers - 1):
                self.convs.append(
                    ModuleList([
                        GATConv(hidden_channels, hidden_channels // heads,
                                heads, add_self_loops=False)
                        for _ in range(num_relations)
                    ]))

        elif self.model == 'rgraphsage':
            self.convs.append(
                ModuleList([
                    SAGEConv(in_channels, hidden_channels, root_weight=False)
                    for _ in range(num_relations)
                ]))

            for _ in range(num_layers - 1):
                self.convs.append(
                    ModuleList([
                        SAGEConv(hidden_channels, hidden_channels,
                                 root_weight=False)
                        for _ in range(num_relations)
                    ]))

        for _ in range(num_layers):
            self.norms.append(BatchNorm1d(hidden_channels))

        self.skips.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.skips.append(Linear(hidden_channels, hidden_channels))

        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(inplace=True),
            Dropout(p=self.dropout),
            Linear(hidden_channels, out_channels),
        )

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x: Tensor, adjs_t: List[SparseTensor]) -> Tensor:
        for i, adj_t in enumerate(adjs_t):
            x_target = x[:adj_t.size(0)]

            out = self.skips[i](x_target)
            for j in range(self.num_relations):
                edge_type = adj_t.storage.value() == j
                subadj_t = adj_t.masked_select_nnz(edge_type, layout='coo')
                if subadj_t.nnz() > 0:
                    out += self.convs[i][j]((x, x_target), subadj_t)

            x = self.norms[i](out)
            x = F.elu(x) if self.model == 'rgat' else F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.mlp(x)

    def training_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        train_loss = F.cross_entropy(y_hat, batch.y)
        self.train_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        self.val_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        self.test_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.25)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default='rgat',
                        choices=['rgat', 'rgraphsage'])
    parser.add_argument('--sizes', type=str, default='25-15')
    parser.add_argument('--in-memory', action='store_true')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    args.sizes = [int(i) for i in args.sizes.split('-')]
    print(args)

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
