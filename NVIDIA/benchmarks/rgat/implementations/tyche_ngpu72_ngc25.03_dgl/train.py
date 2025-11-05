# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# to debug dgl dataloader device=cuda issue.
import os

os.environ['DGL_PREFETCHER_TIMEOUT'] = str(300)

import argparse, datetime
import dgl
import sklearn.metrics
import torch, torch.nn as nn, torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import time, tqdm, numpy as np
from socket import gethostname, gethostbyname
import pylibwholegraph.torch as wgth
import copy
from mlperf_logging import mllog
import warnings

from dllogger import Verbosity
from utility.logger import IntegratedLogger, mllogger
from dataset import Features, IGBHeteroGraphStructure
from dataloading import build_graph, get_loader
from model import get_model, get_feature_extractor, gen_synthetic_block
from apex.optimizers import FusedAdam
from apex.contrib.optimizers.distributed_fused_adam import DistributedFusedAdam
import math
import yaml
import gc
from common.helper import get_offsets_indices
from common.multi_tensor_copy_pad import multi_tensor_copy_pad
warnings.filterwarnings("ignore")

# options, sort by least logs -> most logs
#     "error": 1
#     "warn": 2
#     "info": 3 # default one before wm_log_level commit
#     "debug": 4
#     "trace": 5
WG_LOG_LEVEL="warn" 

def convert_hmmss_to_seconds(hmmss):
    h, mm, ss = [int(x) for x in hmmss.split(":")]
    return ss + mm * 60 + h * 3600

def parse_args():
    parser = argparse.ArgumentParser()
    # Loading dataset
    # path: the root dir that hosts ALL preprocessed IGBH data
    parser.add_argument('--path', type=str, default='/data', 
        help='path containing ALL the WholeGraph preprocessed dataset')
    parser.add_argument("--graph_load_path", type=str, default="/graph", 
        help="path containing preprocessed graph path, for better performance")
    parser.add_argument('--dataset_size', type=str, default='small',
                        choices=['tiny', 'small', 'medium', 'large', 'full'],
                        help='size of the dataset')
    parser.add_argument('--num_classes', type=int, default=19,
                        choices=[19, 2983], help='number of classes')

    # current train script only supports WholeGraph training. 
    parser.add_argument('--wg_sharding_partition', type=str, default='global', choices=['local', 'node', 'global'],
                        help='the domain to shard')
    parser.add_argument('--wg_sharding_location', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='where to shard the embeddings')
    parser.add_argument('--wg_sharding_type', type=str, default='continuous', choices=['continuous', 'chunk', 'distributed'],
                        help='how to partition the embedding')
    parser.add_argument('--concat_embedding_mode', type=str, default=None, 
                        choices=['online', 'offline'], 
                        help='Fuse all the embeddings into one')
    parser.add_argument('--embed_block_size', type=int, default=1)
    parser.add_argument('--num_embed_bucket', type=int, default=4096)
    parser.add_argument('--wg_gather_sm', type=int, default=-1, help='to control the block size when doing embedding lookup')

    # Model
    parser.add_argument('--model_path', type=str, default='/workspace/model.pt')
    parser.add_argument('--model_save', action="store_true")
    parser.add_argument('--amp', action="store_true")
    parser.add_argument('--dist_adam', action="store_true")
    parser.add_argument('--dgl_native_sampler', action="store_true")

    # Model parameters
    parser.add_argument('--fan_out', type=str, default='5,10,15')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument("--validation_batch_size", type=int, default=4096)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--decay', type=float, default=0) # finalized - 0
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument('--log_every', type=int, default=100,
                        help="Display training stats every number of iterations")

    parser.add_argument(
        '--eval_frequency', 
        type=float, default=0.2,
        help="Perform in-epoch evaluation every <eval_frequency> train dataset")
    parser.add_argument(
        '--target_accuracy',
        type=float, default=0.72
    )
    parser.add_argument("--continue_training", action="store_true")

    parser.add_argument("--use_pytorch_dataloader", action="store_true")

    parser.add_argument(
        "--backend", 
        type=str, default="DGL",
        help="Controls the backend for Model / Sampler, and possibly DataLoader for PyG")
    parser.add_argument(
        "--gatconv_backend", 
        type=str, default="native", choices=['native', 'cugraph'],
        help="Controls the backend for GATConv")
    parser.add_argument(
        "--cugraph_switches", 
        type=str, default="1100",
        help="to control [high_precision_wgrad, high_precision_dgrad, deterministic_wgrad, \
            deterministic_dgrad] in cuGraph MHA")
    parser.add_argument('--pad_node_count_to', type=int, default=-1,
                        help="pad the number of input nodes to GATConv. Only effective for cugraph \
                            backend")
    parser.add_argument('--gc_threshold_multiplier', type=int, default=1,
                        help="multiplier factor for increasing the threshold of doing GC")

    parser.add_argument("--internal_results", action="store_true")
    parser.add_argument("--debug_logging", action="store_true")

    parser.add_argument(
        "--sampling_device", 
        type=str, default="cuda",
        choices=['cpu', 'cuda'],
        help="Choose which device to do the sampling")
    
    parser.add_argument(
        "--graph_device", 
        type=str, default="cpu",
        choices=['cpu', 'cuda'],
        help="Choose where the graph is stored")
    
    parser.add_argument(
        "--graph_sharding_partition", 
        type=str, default="node",
        choices=['node', 'global'],
        help="Choose how the graph is sharded")
    
    parser.add_argument('--num_sampling_threads', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument(
        '--seed', default=0, type=int, 
        help="Controls the seed being used for sampling & training"
    )

    # API Logging arguments
    parser.add_argument(
        '--limit_train_batches', default=-1, type=int,
        help="Limit number of train batches in an epoch, -1 for no limit"
    )

    parser.add_argument(
        "--limit_eval_batches", default=-1, type=int, 
        help="Limit number of eval batches in an epoch, -1 for no limit"
    )

    # Overlap sampling and embedding reading
    parser.add_argument("--train_overlap", action="store_true")
    parser.add_argument("--eval_overlap", action="store_true")
    parser.add_argument("--high_priority_embed_stream", action="store_true")

    # Debug related flags
    parser.add_argument(
        "--repeat_input_after", default=-1, type=int, 
        help="Repeat the sampled graph the sampler after certain iterations. Currently only work \
        for overlapped dataloader, and will be ignored otherwise."
    )
    parser.add_argument("--skip_embedding_init", action="store_true",
                        help="Not initialize embedding tables with pre-generated weights")

    # FP8
    parser.add_argument("--fp8_embedding", action="store_true")

    # CUDA Graphs
    parser.add_argument("--use_cuda_graph", action="store_true")
    parser.add_argument('--cuda_graph_estimation_batches', type=int, default=20)
    parser.add_argument("--cuda_graph_padding_sigma", type=int, default=3)
    parser.add_argument("--dump_cuda_graph_shape_info", action="store_true")
    parser.add_argument("--load_cuda_graph_shape_info", action="store_true")
    parser.add_argument("--warmup_model", action="store_true")

    # This script does not have offline partitioning, so we do not add command-line argument for that

    args = parser.parse_args()
    args.num_layers = len(args.fan_out.split(","))

    if args.fp8_embedding: 
        args.path = f"{args.path}/float8"
    else:
        args.path = f"{args.path}/float16"
    
    ### compatibility check
    if args.backend != "DGL":
        if args.gatconv_backend == "cugraph":
            raise NotImplementedError("Currently only DGL supports cuGraph GATConv!")
    if args.use_cuda_graph:
        if args.amp == False or args.dist_adam == False:
            raise NotImplementedError("Currently full iteration cuda graph assumes amp and distribtued Adam!")
    return args


def load_dataset(
        # dataset paths
        path, dataset_size, num_classes,
        # backend
        backend,  
        feature_store,

        # wholegraph related
        wholegraph_comms=None,
        graph_device='cpu',
        sampling_device='cuda',
        graph_sharding_partition='node',
        skip_embedding_init=False,
):
    dataset = IGBHeteroGraphStructure(
        config=feature_store.config,
        path=path, dataset_size=dataset_size, num_classes=num_classes,
        wholegraph_comms=wholegraph_comms,
        graph_device=graph_device,
        sampling_device=sampling_device,
        graph_sharding_partition=graph_sharding_partition,
    )
 
    if skip_embedding_init == False:
        feature_store.build_features()

    graph = build_graph(
        graph_structure=dataset,
        features=feature_store,
        backend=backend
    )

    return dataset, feature_store, graph


def load_model_loss_optimizer(
        # general arguments
        etypes, backend, gatconv_backend, switches, pad_node_count_to, device, amp, dist_adam,
        # model specific arguments
        in_feats, h_feats, num_classes, 
        num_layers, n_heads, dropout, with_trim,
        # optimizer specific arguments
        learning_rate, decay
    ):

    model = get_model(
        backend=backend, gatconv_backend=gatconv_backend, switches=switches, 
        pad_node_count_to=pad_node_count_to, etypes=etypes, 
        in_feats=in_feats, h_feats=h_feats, num_classes=num_classes,
        num_layers=num_layers, n_heads=n_heads, dropout=dropout, with_trim=with_trim,
    ).to(device)
    
    if amp:
        model = model.half()

    # cugraph_dgl only requires CSC format, hence convert earlier
    feature_extractor = get_feature_extractor(backend=backend, 
                                              formats=['csc'] if gatconv_backend=='cugraph' 
                                                                    and backend == 'DGL'
                                                              else None)

    loss = nn.CrossEntropyLoss().to(device)

    side_cuda_stream = torch.cuda.Stream()
    side_cuda_stream.wait_stream(torch.cuda.current_stream())
    
    if dist_adam:
        global_rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        world_group = torch.distributed.distributed_c10d._get_default_group()
        self_groups = [torch.distributed.new_group(ranks=[i]) for i in range(world_size)]
        # warm up self group communicator
        torch.distributed.barrier(group=self_groups[global_rank], device_ids=[local_rank])
        
        # grad dtype should follow param dtype
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        param_size_mb = math.ceil(param_size/1024**2)

        side_cuda_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_cuda_stream):
            optimizer = DistributedFusedAdam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=decay,
                overlap_grad_sync=False,
                contiguous_param_buffer=True,
                contiguous_grad_buffer=True,
                distributed_process_group=self_groups[global_rank],
                redundant_process_group=world_group,
                grad_sync_dtype=torch.float16 if amp else torch.float32,
                param_sync_dtype=torch.float16 if amp else torch.float32,
                bucket_cap_mb=param_size_mb,
                capturable=True,
            )
            optimizer.init_params()
            optimizer.init_param_buffer()
            optimizer.zero_grad()
            grad_scaler = torch.cuda.amp.GradScaler(init_scale=1024) if amp else None

        torch.cuda.current_stream().wait_stream(side_cuda_stream)
    else:
        with torch.cuda.stream(side_cuda_stream):
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device],
                output_device=device,
                find_unused_parameters=False,
                static_graph=True,
            )
            for attribute in model.module.attributes_to_register:
                setattr(model, attribute, getattr(model.module, attribute))

        torch.cuda.current_stream().wait_stream(side_cuda_stream)

        if amp:
            optimizer = FusedAdam(
                params=model.parameters(),
                lr=learning_rate,
                weight_decay=decay,
                master_weights=True,
                capturable=True,
            )
            grad_scaler = torch.cuda.amp.GradScaler(init_scale=1024)
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=decay
            )
            grad_scaler = None

    return model, feature_extractor, loss, optimizer, grad_scaler, side_cuda_stream


################
# CUDA Graph Utils
################



def check_cuda_graph_eligible(
    actual_blocks_offsets_indices, 
    actual_features, 
    actual_labels,
    static_blocks_offsets_indices, 
    static_features,
    static_labels
):
    # last incomplete batch will have different label shape
    if static_labels.shape[0] != actual_labels.shape[0]:
        return False

    for ntype in static_features:
        if static_features[ntype].shape[0] < actual_features[ntype].shape[0]:
            return False
        
    for (static_offset_index, dynamic_offset_index) in zip(static_blocks_offsets_indices, actual_blocks_offsets_indices):
        for etype in static_offset_index:
            static_offset, static_index = static_offset_index[etype]
            dynamic_offset, dynamic_index = dynamic_offset_index[etype]

            if dynamic_offset.numel() > static_offset.numel():
                return False

            if dynamic_index.numel() > static_index.numel():
                return False
    
    return True


def capture_graph(capture_stream, blocks_static, in_feats, device, model, loss_fn, grad_scaler, optimizer):
    # warmup
    x_static = {node: torch.randn((blocks_static[0].num_src_nodes(node), in_feats), device=device, dtype=torch.half) for node in blocks_static[0].ntypes}
    label_static = torch.zeros(blocks_static[-1].num_dst_nodes('paper'), device=device, dtype=torch.int64)
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):  
        for _ in range(11):    
            optimizer.zero_grad()
            batch_pred_static = model(blocks_static, x_static)     
            loss_static = loss_fn(batch_pred_static, label_static)
            grad_scaler.scale(loss_static).backward()
            # to make sure warmup doesn't alter the state of the optimizer
            optimizer.zero_grad()
            grad_scaler.step(optimizer)
            grad_scaler.update()
    torch.cuda.current_stream().wait_stream(capture_stream)
    # recover the step of the optimizer
    optimizer.state["step"].fill_(0)

    # capture
    g = torch.cuda.CUDAGraph()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(g, stream=capture_stream):
        optimizer.zero_grad()
        batch_pred_static = model(blocks_static, x_static)
        loss_static = loss_fn(batch_pred_static, label_static)
        grad_scaler.scale(loss_static).backward()
    torch.cuda.current_stream().wait_stream(capture_stream)

    g_optim = torch.cuda.CUDAGraph()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(g_optim, stream=capture_stream):
        grad_scaler.step(optimizer)
        grad_scaler.update()
    torch.cuda.current_stream().wait_stream(capture_stream)

    # extract static buffer for offsets and indices
    list_satic_offsets_indices = get_offsets_indices(blocks_static)
    return g, g_optim, list_satic_offsets_indices, x_static, label_static, batch_pred_static, loss_static


def prepare_static_buffer(list_static_offsets_indices, x_static, label_static, block_offsets_indices, x, label):
    with torch.no_grad():
        for static_offset_indices, dynamic_offset_indices in zip(list_static_offsets_indices, block_offsets_indices):
            for relation in static_offset_indices:
                static_offset, static_indices = static_offset_indices[relation]
                dynamic_offset, dynamic_indices = dynamic_offset_indices[relation]

                # handle padding
                assert static_offset.numel() >= dynamic_offset.numel()
                assert static_indices.numel() >= dynamic_indices.numel()
                static_offset[:dynamic_offset.numel()] = dynamic_offset
                static_offset[dynamic_offset.numel():] = dynamic_offset[-1]
                static_indices[:dynamic_indices.numel()] = dynamic_indices


        for ntype in x_static:
            row = x[ntype].shape[0]
            # handle padding
            assert row <= x_static[ntype].shape[0]
            
            x_static[ntype][:row] = x[ntype]
            x_static[ntype][row:] = 0

        label_static.copy_(label)

def prepare_static_buffer_triton(list_static_offsets_indices, x_static, label_static, block_offsets_indices, x, label):
    with torch.no_grad():
        list_src = []
        list_dst = []
        for static_offset_indices, dynamic_offset_indices in zip(list_static_offsets_indices, block_offsets_indices):
            for relation in static_offset_indices:
                static_offset, static_indices = static_offset_indices[relation]
                dynamic_offset, dynamic_indices = dynamic_offset_indices[relation]
                list_src.append(dynamic_offset)
                list_src.append(dynamic_indices)
                list_dst.append(static_offset)
                list_dst.append(static_indices)

        multi_tensor_copy_pad(list_src, list_dst, BLOCK_SIZE=256)

        for ntype in x_static:
            row = x[ntype].shape[0]
            # handle padding
            assert row <= x_static[ntype].shape[0]
            
            x_static[ntype][:row] = x[ntype]
            x_static[ntype][row:] = 0

        label_static.copy_(label)


def estimate_cuda_graph_input_shapes(loader, num_samples_for_estimation, n_sigma=3):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    total_spaces_needed = world_size * num_samples_for_estimation

    node_counts = []
    edge_counts = []

    defined_batch_size = loader.batch_size

    for idx, actual_batch in enumerate(loader):
        batch_size = actual_batch[-1][-1].label.shape[0]

        if batch_size != defined_batch_size:
            assert idx != (len(loader)-1), f"{batch_size} != {defined_batch_size}"
            # we do not use data from incomplete batches, otherwise std will increase. 
            # continue

        current_batch_idx = rank * num_samples_for_estimation + idx

        if idx >= num_samples_for_estimation:
            break
        _, _, blocks = actual_batch
        for layer, block in enumerate(blocks):
            while len(node_counts) <= layer:
                node_counts.append({})

            while len(edge_counts) <= layer:
                edge_counts.append({})

            for ntype in block.srctypes:
                if ntype not in node_counts[layer]:
                    node_counts[layer][ntype] = torch.zeros(total_spaces_needed).to(block.device)
                
                node_counts[layer][ntype][current_batch_idx] += block.num_src_nodes(ntype)

            for etype in block.canonical_etypes:
                if etype not in edge_counts[layer]:
                    edge_counts[layer][etype] = torch.zeros(total_spaces_needed).to(block.device)

                edge_counts[layer][etype][current_batch_idx] += block.num_edges(etype)

    max_node_counts = []
    max_edge_counts = []
    for layer in range(len(node_counts)):
        while len(max_node_counts) <= layer:
            max_node_counts.append({})
        
        while len(max_edge_counts) <= layer:
            max_edge_counts.append({})

        for ntype in node_counts[layer]:
            dist.all_reduce(node_counts[layer][ntype], op=dist.ReduceOp.SUM)
            mean = node_counts[layer][ntype].mean()
            std = node_counts[layer][ntype].std()
            max_node_counts[layer][ntype] = math.ceil((mean + n_sigma * std).cpu().item())

        for etype in edge_counts[layer]:
            dist.all_reduce(edge_counts[layer][etype], op=dist.ReduceOp.SUM)
            mean = edge_counts[layer][etype].mean()
            std = edge_counts[layer][etype].std()
            max_edge_counts[layer][etype] = math.ceil((mean + n_sigma * std).cpu().item())
    return max_node_counts, max_edge_counts

def warmup_model(model, batch_size, in_feats, device):
    warmup_size = batch_size
    list_edge_type = list(model.layers[0].mod_dict.keys())
    list_node_type = list(set([x[0] for x in list_edge_type]))
    num_layers = len(model.layers)
    list_node_counts, list_edge_counts = [], []
    for layer in range(num_layers):
        list_node_counts.append({ntype: warmup_size for ntype in list_node_type})
        if layer < num_layers - 1:
            list_edge_counts.append({etype: warmup_size for etype in list_edge_type})
        else:
            list_edge_counts.append({etype: warmup_size if etype[2] == 'paper' else 0 for etype in list_edge_type})
    list_node_counts.append({ntype: batch_size if ntype == 'paper' else 0 for ntype in list_node_type})
    blocks_warmup = gen_synthetic_block(list_node_counts, list_edge_counts, batch_size=batch_size, device=device)
    x_warmup = {node: torch.randn((blocks_warmup[0].num_src_nodes(node), in_feats), device=device, dtype=torch.half) for node in blocks_warmup[0].ntypes}
    y = model(blocks_warmup, x_warmup)
    loss = (y * 0).mean()
    loss.backward()


def train(
        device, 
        model, feature_extractor, 
        loss_fcn, optimizer, amp, grad_scaler,
        dataloader, features,
        logger, log_every, 
        # in-epoch evaluation controllers
        eval_interval, eval_dataloader, target_accuracy, continue_training,
        limit_train_batches, limit_eval_batches, current_epoch,
        train_overlap, eval_overlap,
        graph_related):
    # Loop over the dataloader to sample the computation dependency graph as a list of blocks.

    world_size = dist.get_world_size()
    epoch_loss = 0
    epoch_acc = 0
    epoch_start = time.time()
    logger.debug(f"Toggling model as train here.")
    model.train()

    success = False

    gpu_mem_alloc = []

    logger.debug("Making the dataloader enumerable here")
    iterator = enumerate(dataloader)

    logger.debug(f"Starting to iterate through the dataloader")
    step_time_accumulator = []
    last_logging_acc = 0
    last_logging_loss = 0
    total_steps = 0

    batch_num = len(dataloader)

    model_graphed, optimizer_graphed, list_static_offsets_indices, x_static, label_static, batch_pred_static, loss_static = graph_related
    step_start = time.time()
    for step, batch in iterator:
        if limit_train_batches >= 0 and step >= limit_train_batches:
            break
        total_steps = step

        if train_overlap:
            actual_batch, batch_inputs, batch_labels = dataloader.get_inputs_and_outputs()
        else:
            actual_batch = feature_extractor.extract_graph_structure(batch, device)
            batch_inputs, batch_labels = feature_extractor.extract_inputs_and_outputs(
                actual_batch,
                device,
                features
            )

        if not amp:
            batch_inputs_tmp = {
                key: value.float()
                for key, value in batch_inputs.items()
                }
            batch_inputs = batch_inputs_tmp
        
        used_cuda_graph = False

        if model_graphed is not None:
            if train_overlap:
                dynamic_offsets_indices = dataloader.get_dynamic_offsets_indices()
            else:
                dynamic_offsets_indices = get_offsets_indices(actual_batch)
            if check_cuda_graph_eligible(dynamic_offsets_indices, batch_inputs, batch_labels, list_static_offsets_indices, x_static, label_static):
                prepare_static_buffer_triton(list_static_offsets_indices, x_static, label_static, dynamic_offsets_indices, batch_inputs, batch_labels)
                model_graphed.replay()
                used_cuda_graph = True
            else:
                batch_pred = model(actual_batch, batch_inputs)
                loss = loss_fcn(batch_pred, batch_labels)
                optimizer.zero_grad()
                if amp: 
                    loss_scaled = grad_scaler.scale(loss)
                    loss_scaled.backward()
                else:
                    loss.backward()
        else:
            batch_pred = model(actual_batch, batch_inputs)

            logger.debug(f"    On step {step}, Model forward complete. Calculating loss.", do_log=(step < 2))

            loss = loss_fcn(batch_pred, batch_labels)

            logger.debug(f"    On step {step}, Loss calculated. Backprop and optimizer step.", do_log=(step < 2))

            optimizer.zero_grad()
            if amp:
                loss_scaled = grad_scaler.scale(loss)
                loss_scaled.backward()
            else:
                loss.backward()

        if train_overlap:
            dataloader.check_overlap_launch_end()
        if amp:
            if used_cuda_graph:
                optimizer_graphed.replay()
            else:
                grad_scaler.step(optimizer)
                grad_scaler.update()
        else:
            optimizer.step()

        logger.debug(f"    On step {step}, Backprop and optimizer step complete. Gathering metrics. ",
                     do_log=(step < 2))
        loss_display = loss_static if used_cuda_graph else loss
        batch_pred_display = batch_pred_static if used_cuda_graph else batch_pred
        epoch_loss += loss_display.detach().item()
        epoch_acc += sklearn.metrics.accuracy_score(batch_labels.cpu().numpy(),
                                                   batch_pred_display.argmax(1).detach().cpu().numpy())
        gpu_mem_alloc.append(
            torch.cuda.max_memory_allocated(device=device) / 1000000
            if torch.cuda.is_available()
            else 0
        )
        step_time_accumulator.append((time.time() - step_start) * 1000)

        if step % log_every == 0:
            logger.log(
                step=f'Iteration {step}',
                data={
                    "train_acc": (epoch_acc - last_logging_acc) / log_every,
                    "iteration_time": sum(step_time_accumulator) / len(step_time_accumulator),
                    "train_loss": (epoch_loss - last_logging_loss) / log_every,
                    "last_iter_used_cuda_graph": used_cuda_graph
                },
                verbosity=(Verbosity.DEFAULT)
            )
            last_logging_loss = epoch_loss
            last_logging_acc = epoch_acc
            step_time_accumulator = []

        if (step + 1) % eval_interval == 0: 

            model.eval()

            epoch_number = round(current_epoch + step / batch_num, 2)
            
            val_acc, val_time, val_steps = evaluate(
                device=device, 
                model=model,
                feature_extractor=feature_extractor,
                amp=amp,
                dataloader=eval_dataloader,
                features=features,
                logger=logger,
                limit_eval_batches=limit_eval_batches, 
                current_epoch_num=epoch_number,
                world_size=world_size,
                eval_overlap=eval_overlap)

            model.train()

            logger.log(
                step=f"Iteration {step}",
                data={
                    "val_acc": val_acc,
                    "val_time": val_time, 
                    "val_steps": val_steps
                }
            )

            if val_acc > target_accuracy:
                success = True
                if not continue_training:
                    break
        step_start = time.time()

    if train_overlap:
        dataloader.thread_cleanup()

    epoch_acc /= total_steps
    epoch_gpu_mem = sum(gpu_mem_alloc) / len(gpu_mem_alloc)
    train_time = str(datetime.timedelta(seconds=int(time.time() - epoch_start)))
    
    return epoch_acc, epoch_loss, epoch_gpu_mem, train_time, total_steps, success, round(current_epoch + total_steps / batch_num, 2)


def evaluate(
        device, 
        model, feature_extractor, amp,
        dataloader, features,
        logger,
        limit_eval_batches,
        current_epoch_num, world_size,
        eval_overlap):
    
    # train_rgnn_multi_gpu.py line 39: eval_start within the evaluate function
    mllogger.start(
        key=mllogger.constants.EVAL_START, 
        value=current_epoch_num, 
        sync=False, 
        metadata={mllogger.constants.EPOCH_NUM: current_epoch_num}
    )

    # adding logger here for potential debugging need
    model.eval()
    epoch_start = time.time()
    predictions = []
    labels = []
    with torch.no_grad():
        # for a more verbose evaluation
        # iterator = tqdm.tqdm(dataloader) 
        for step, batch in enumerate(dataloader):
            if limit_eval_batches >= 0 and step >= limit_eval_batches:
                break
            if eval_overlap:
                actual_batch, batch_inputs, batch_labels = dataloader.get_inputs_and_outputs()
            else:
                actual_batch = feature_extractor.extract_graph_structure(batch, device)
                batch_inputs, batch_labels = feature_extractor.extract_inputs_and_outputs(
                    actual_batch, device, features
                )
            if not amp:
                batch_inputs_tmp = {
                    key: value.float()
                    for key, value in batch_inputs.items()
                    }
                batch_inputs = batch_inputs_tmp
            
            batch_preds= model(actual_batch, batch_inputs)

            if eval_overlap:
                dataloader.check_overlap_launch_end()

            labels.append(batch_labels.cpu().numpy())
            predictions.append(batch_preds.argmax(1).cpu().numpy())

        logger.log(
            step=f'Current_epoch_num {current_epoch_num}',
            data={
                "eval_epoch_time": (time.time() - epoch_start) * 1000,
                "total_steps": step,
            },
            verbosity=(Verbosity.DEFAULT)
        )

        if eval_overlap:
            dataloader.thread_cleanup()

        total_steps = len(predictions)
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        acc = sklearn.metrics.accuracy_score(labels, predictions)

    # train_rgnn_multi_gpu.py line 65: performs the global all_reduce before eval_stop
    all_reduced_val_acc = torch.tensor(acc).to(device)

    dist.all_reduce(all_reduced_val_acc, op=dist.ReduceOp.SUM)
    all_reduced_val_acc /= world_size

    all_reduced_val_acc = all_reduced_val_acc.to("cpu").item()

    mllogger.event(
        key=mllog.constants.EVAL_ACCURACY, 
        value=all_reduced_val_acc, 
        metadata={mllogger.constants.EPOCH_NUM: current_epoch_num}
    )

    mllogger.end(
        key=mllogger.constants.EVAL_STOP, 
        value=current_epoch_num,
        sync=False, 
        metadata={mllogger.constants.EPOCH_NUM: current_epoch_num}
    )

    model.train()
    return (all_reduced_val_acc, str(datetime.timedelta(seconds=int(time.time() - epoch_start))), total_steps)


def main():
    mllogger.start(key=mllog.constants.INIT_START, unique=True, unique_log_rank=0)
    args = parse_args()

    # fixing seeds here
    SEED = args.seed
    torch.manual_seed(SEED)
    dgl.seed(SEED)
    dgl.random.seed(SEED)
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    embedding_tensor_dict = None
    embedding_comms = None
    default_embedding_tensor_dict = None

    if "SLURM_NTASKS_PER_NODE" in os.environ:
        node_size = int(os.environ.get("SLURM_NTASKS_PER_NODE"))
    elif "LOCAL_WORLD_SIZE" in os.environ:
        node_size = int(os.environ.get("LOCAL_WORLD_SIZE"))
    else:
        raise Exception("Local world size is not specified")

    # dist.initialize_process_group("nccl") is contained here
    wgth.init_torch_env(global_rank,
                        world_size,
                        local_rank,
                        node_size,
                        WG_LOG_LEVEL)
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.distributed.barrier()
    # Prevent following communicators to lock the tree
    os.environ["NCCL_SHARP_DISABLE"] = "1"
    os.environ["NCCL_COLLNET_ENABLE"] = "0"
    
    logger = IntegratedLogger(
        proc_id=torch.distributed.get_rank(),
        internal_results=args.internal_results,
        debug_logging=args.debug_logging,
        print_only=True
    )
    
    # We log this after torch initialization since otherwise they will be reported in every ranks
    mllogger.mlperf_submission_log(benchmark=mllog.constants.RGAT)
    mllogger.event(key=mllog.constants.SEED, value=args.seed, unique=True, unique_log_rank=0)

    # WholeGraph pre dataset loading setups: set default arguments

    embedding_tensor_dict = {
        node_name: {
            "partition": args.wg_sharding_partition, 
            "location": args.wg_sharding_location,  
            "type": args.wg_sharding_type, 
        }
        for node_name in ['paper', 'author', 'institute', 'fos', 'journal', 'conference']
    }
    
    # WholeGraph inter-node/GPU communications
    embedding_comms = {
        "node": wgth.get_local_node_communicator(),
        "global": wgth.get_global_communicator(),
        "local": wgth.get_local_device_communicator()
    }

    # Init WG storage
    logger.log(step="Init", data={'message': f"Start create feature storage"}, verbosity=Verbosity.DEFAULT)
    feature_store = Features(
        path=args.path, dataset_size=args.dataset_size,
        embedding_tensor_dict=embedding_tensor_dict, 
        wholegraph_comms=embedding_comms,
        concat_embedding_mode=args.concat_embedding_mode,
        wg_gather_sm=args.wg_gather_sm,
        fp8_embedding = args.fp8_embedding,
    )
    logger.log(step="Init", data={'message': f"Start warming up feature featch"}, verbosity=Verbosity.DEFAULT)
    feature_store.warm_up()
    warmup_t = torch.zeros(1, device=device)
    dist.all_reduce(warmup_t)
    del warmup_t

    # model init should happen within the untimed region
    logger.log(step="Init", data={'message': f"Start loading model"}, verbosity=Verbosity.DEFAULT)
    model, feature_extractor, loss, optimizer, grad_scaler, side_stream = load_model_loss_optimizer(
        etypes=feature_store.edge_types, backend=args.backend, 
        gatconv_backend=args.gatconv_backend, switches=args.cugraph_switches, pad_node_count_to=args.pad_node_count_to,
        device=device, amp=args.amp, dist_adam=args.dist_adam,
        in_feats=1024, h_feats=args.hidden_channels, num_classes=args.num_classes,
        num_layers=args.num_layers, n_heads=args.num_heads, dropout=args.dropout, with_trim=False,
        learning_rate=args.learning_rate, decay=args.decay,
    )
    if args.warmup_model:
        logger.log(step="Init", data={'message': f"Start warming up model"}, verbosity=Verbosity.DEFAULT)
        warmup_model(model=model, batch_size=args.batch_size, in_feats=1024, device=device)

    logger.log(step="Init", data={'message': f"Start touching the dataset"}, verbosity=Verbosity.DEFAULT)
    mllogger.event(key=mllog.constants.GLOBAL_BATCH_SIZE, value=args.batch_size * world_size, unique=True, unique_log_rank=0)
    mllogger.event(key=mllog.constants.GRADIENT_ACCUMULATION_STEPS, value=1, unique=True, unique_log_rank=0)
    mllogger.event(key=mllog.constants.OPT_NAME, value="adam", unique=True, unique_log_rank=0)
    mllogger.event(key=mllog.constants.OPT_BASE_LR, value=args.learning_rate, unique=True, unique_log_rank=0)
    mllogger.log_init_stop_run_start()

    if args.gc_threshold_multiplier != 1:
        thresholds = gc.get_threshold()
        gc.set_threshold(
            thresholds[0] * args.gc_threshold_multiplier,
            thresholds[1] * args.gc_threshold_multiplier,
            thresholds[2] * args.gc_threshold_multiplier,
        )

    # we touch the dataset here
    dataset, feature_store, graph = load_dataset(
        path=args.graph_load_path, dataset_size=args.dataset_size, num_classes=args.num_classes,
        backend=args.backend, feature_store=feature_store,
        wholegraph_comms=embedding_comms,
        graph_device=args.graph_device,
        sampling_device=args.sampling_device,
        graph_sharding_partition=args.graph_sharding_partition,
        skip_embedding_init=args.skip_embedding_init,
    )

    mllogger.event(key=mllog.constants.TRAIN_SAMPLES, value=dataset.train_indices.size(0), unique=True, unique_log_rank=0)
    mllogger.event(key=mllog.constants.EVAL_SAMPLES, value=dataset.val_indices.size(0), unique=True, unique_log_rank=0)

    logger.log(step="Init", data={'message': f"Done init"}, verbosity=Verbosity.DEFAULT)

    if args.sampling_device == 'cuda':
        graph = graph.to(device)
        dataset.train_indices = dataset.train_indices.to(device)
        dataset.val_indices = dataset.val_indices.to(device)

    logger.debug(f"Group of size {dist.get_world_size()} initialized? {dist.is_initialized()}", do_log=True)

    logger.debug(f"Process group initialized, initializing sampler through dgl.dataloading.MultiLayerNeighborSampler.")
    train_idx = dataset.train_indices.split(
        dataset.train_indices.size(0) // world_size
    )[global_rank]
    val_idx = dataset.val_indices.split(
        dataset.val_indices.size(0) // world_size
    )[global_rank]

    # we start estimating the max block placeholder size here
    if args.use_cuda_graph:
        logger.log(step="Init", data={'message': f"Getting the loader for estimate statiscs"}, verbosity=Verbosity.DEFAULT)
        cuda_graph_estimator_loader = get_loader(
            graph=graph,
            index=train_idx, 
            fanouts=args.fan_out, 
            backend=args.backend,
            use_torch=args.use_pytorch_dataloader,
            num_sampling_threads=args.num_sampling_threads,
            pyg_style_sampler=not args.dgl_native_sampler,
            enable_overlap=False, # No overlap - we only need sampling results anyway
            feature_extractor=None,
            features=None,
            repeat_input_after=args.repeat_input_after,
            high_priority_embed_stream=args.high_priority_embed_stream, 

            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            device=device,
            num_workers=args.num_workers
        )
        logger.log(step="Init", data={'message': f"Start estimating the statistics"}, verbosity=Verbosity.DEFAULT)
        max_node_counts, max_edge_counts = estimate_cuda_graph_input_shapes(
            cuda_graph_estimator_loader, 
            num_samples_for_estimation=args.cuda_graph_estimation_batches, 
            n_sigma=args.cuda_graph_padding_sigma
        )
        if args.dump_cuda_graph_shape_info:
            max_edge_counts_ = [{'__'.join(key): value for key, value in d.items()} for d in max_edge_counts]
            with open(f'cuda_graph_shape_info_sigma{args.cuda_graph_padding_sigma}_bs{args.batch_size}.yaml', 'w') as yaml_file:
                yaml.dump([max_node_counts, max_edge_counts_], yaml_file, default_flow_style=False)
        if args.load_cuda_graph_shape_info:
            with open(f'cuda_graph_shape_info_sigma{args.cuda_graph_padding_sigma}_bs{args.batch_size}.yaml', 'r') as yaml_file:
                max_node_counts, max_edge_counts_ = yaml.safe_load(yaml_file)
                max_edge_counts = [{tuple(key.split('__')): value for key, value in d.items()} for d in max_edge_counts_]
        logger.log(step="Init", data={'message': f"Start generating synthetic blocks"}, verbosity=Verbosity.DEFAULT)
        blocks_placeholder = gen_synthetic_block(max_node_counts, max_edge_counts, batch_size=args.batch_size, device=device)
    else:
        blocks_placeholder = None

    logger.log(step="Init", data={'message': f"Start capturing CUDA graph"}, verbosity=Verbosity.DEFAULT)
    if args.use_cuda_graph:
        model_graphed, optimizer_graphed, list_static_offsets_indices, x_static, label_static, batch_pred_static, loss_static = capture_graph(side_stream, blocks_placeholder, 1024, device, model, loss, grad_scaler, optimizer)
    else:
        model_graphed, optimizer_graphed, list_static_offsets_indices, x_static, label_static, batch_pred_static, loss_static = None, None, None, None, None, None
    
    logger.debug(f"Train val test data got. Initializing train dataloader through dgl.dataloading.DataLoader.")
    logger.log(step="Init", data={'message': f"Start init dataloaders"}, verbosity=Verbosity.DEFAULT)
    train_dataloader = get_loader(
        graph=graph,
        index=train_idx,
        fanouts=args.fan_out,
        backend=args.backend,
        use_torch=args.use_pytorch_dataloader,
        num_sampling_threads=args.num_sampling_threads,
        pyg_style_sampler=not args.dgl_native_sampler,
        enable_overlap=args.train_overlap,
        feature_extractor=feature_extractor,
        features=feature_store,
        repeat_input_after=args.repeat_input_after,
        high_priority_embed_stream=args.high_priority_embed_stream,
        pre_get_dynamic_offsets_indices=args.use_cuda_graph,

        # **kwargs
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False, 
        device=device,
        num_workers=args.num_workers,
    )

    logger.debug(f"Train dataloader initialized. Initializing valid dataloader through dgl.dataloading.DataLoader.")

    val_dataloader = get_loader(
        graph=graph,
        index=val_idx,
        fanouts=args.fan_out,
        backend=args.backend,
        use_torch=args.use_pytorch_dataloader,
        num_sampling_threads=args.num_sampling_threads,
        pyg_style_sampler=not args.dgl_native_sampler,
        enable_overlap=args.eval_overlap,
        feature_extractor=feature_extractor,
        features=feature_store,
        repeat_input_after=args.repeat_input_after,
        high_priority_embed_stream=args.high_priority_embed_stream,

        # **kwargs
        batch_size=args.validation_batch_size,
        shuffle=False,
        drop_last=False, 
        device=device,
        num_workers=args.num_workers,
    )

    eval_every_n_training_steps = int(len(train_dataloader) * args.eval_frequency)

    logger.debug(f"Valid dataloader initialized. Initializing test dataloader through dgl.dataloading.DataLoader.")
    logger.log(step="Init", data={'message': f"Done init dataloaders"}, verbosity=Verbosity.DEFAULT)

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    logger.metadata("model_size", {"unit": "MB", "format": ":.3f"})
    hp_config = {"model_size": size_all_mb}
    hp_config.update(vars(args))
    logger.log(step="PARAMETER", data=hp_config, verbosity=Verbosity.DEFAULT)

    logger.metadata("gpu_memory", {"unit": "MB", "format": ":.2f"})
    logger.metadata("train_loss", {"unit": "", "format": ":.4f"})
    logger.metadata("train_acc", {"unit": "%", "format": ":.4f"})
    logger.metadata("valid_acc", {"unit": "%", "format": ":.4f"})
    logger.metadata("iteration_time", {"unit": "ms", "format": ":.2f"})
    logger.metadata("best_valid_acc", {"unit": "%", "format": ":.2f"})
    logger.metadata("eval_epoch_time", {"unit": "ms", "format": ":.2f"})

    training_start = time.time()

    status = mllogger.constants.ABORTED
    for epoch in range(args.epochs):
        mllogger.start(key=mllogger.constants.EPOCH_START, metadata={mllogger.constants.EPOCH_NUM: epoch+1})
        logger.debug(f"Starting epoch {epoch}")
        acc = torch.tensor([0., ]).to(device)

        epoch_acc, epoch_loss, epoch_gpu_mem, train_time, total_steps, train_success, epoch_number = train(
            device=device, 
            model=model, feature_extractor=feature_extractor,
            loss_fcn=loss, optimizer=optimizer, amp=args.amp, grad_scaler=grad_scaler,
            dataloader=train_dataloader, features=feature_store,
            logger=logger, log_every=args.log_every,
            eval_interval=eval_every_n_training_steps,
            eval_dataloader=val_dataloader,
            target_accuracy=args.target_accuracy,
            continue_training=args.continue_training,
            limit_train_batches=args.limit_train_batches, 
            limit_eval_batches=args.limit_eval_batches,
            current_epoch=epoch,
            train_overlap=args.train_overlap,
            eval_overlap=args.eval_overlap,
            graph_related = (model_graphed, optimizer_graphed, list_static_offsets_indices, x_static, label_static, batch_pred_static, loss_static)
        )

        acc += epoch_acc
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        acc /= world_size

        logger.log(
            step=(epoch, "train"),
            data={
                "train_acc": acc[0].item(),
                "train_time": train_time,
                "train_loss": epoch_loss,
                "gpu_memory": epoch_gpu_mem,
                "total_train_steps": total_steps,
            },
            verbosity=(Verbosity.DEFAULT)
        )

        train_time_in_seconds = convert_hmmss_to_seconds(train_time)

        mllogger.event(key="tracked_stats", value={
            "train_time": round(train_time_in_seconds / 60, 2), # per-epoch train minutes
            # throughput in number of nodes per sec
            "throughput": args.batch_size * total_steps * world_size / train_time_in_seconds,    
            "train_accuracy": acc[0].item(),
            "max_memory_usage": round(torch.cuda.max_memory_allocated()/(1024*1024*1024), 3)
        }, metadata={"epoch_num": (epoch_number)})

        if train_success:
            status = mllogger.constants.SUCCESS
            mllogger.end(key=mllogger.constants.EPOCH_STOP, sync=False, metadata={mllogger.constants.EPOCH_NUM: epoch_number})
            if not args.continue_training:
                break
        else:

            model.eval()
            # if not train success: perform another evaluation at end of epoch
            val_acc, val_time, val_steps = evaluate(
                device=device,
                model=model, feature_extractor=feature_extractor, amp=args.amp,
                dataloader=val_dataloader,
                features=feature_store,
                logger=logger, 
                limit_eval_batches=args.limit_eval_batches,
                current_epoch_num=epoch+1,
                world_size=world_size,
                eval_overlap=args.eval_overlap
            )

            model.train()
            
            logger.log(
                step=(epoch+1, "valid"),
                data={
                    "valid_acc": val_acc,
                    "valid_time": val_time
                },
                verbosity=Verbosity.DEFAULT
            )

            if val_acc >= args.target_accuracy:
                train_success = True

        if train_success:
            status = mllogger.constants.SUCCESS
            mllogger.end(key=mllogger.constants.EPOCH_STOP, sync=False, metadata={mllogger.constants.EPOCH_NUM: epoch_number})
            break

        torch.cuda.synchronize()
        torch.distributed.barrier()

        mllogger.end(key=mllogger.constants.EPOCH_STOP, sync=False, metadata={mllogger.constants.EPOCH_NUM: epoch_number})

    logger.log(
        step=(args.epochs, "final"),
        data={
            "total_time": str(datetime.timedelta(seconds=int(time.time() - training_start)))
        },
        verbosity=Verbosity.DEFAULT
    )

    logger.log(
        step=(),
        data={
            "total_time": str(datetime.timedelta(seconds=int(time.time() - training_start)))
        },
        verbosity=Verbosity.DEFAULT
    )

    mllogger.event(key="tracked_stats", value={
        "walltime": round(int(time.time() - training_start) / 60, 2) # total training minutes
    }, metadata={mllogger.constants.EPOCH_NUM: epoch_number})

    mllogger.log_run_stop(status=status, **{mllog.constants.EPOCH_NUM: epoch_number})
    mllogger.event(key=mllog.constants.STATUS, value=status, metadata={mllog.constants.STATUS: status, mllog.constants.EPOCH_NUM: epoch_number})
    del model_graphed, optimizer_graphed
    wgth.finalize()


if __name__ == "__main__":
    main()
