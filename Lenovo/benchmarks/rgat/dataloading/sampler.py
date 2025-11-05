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

import dgl
import torch

from dgl.heterograph import DGLBlock

def make_block(full_graph, edges, nnodes, dst_nodes):
    relations = []
    for relation in edges:
        relations.append(dgl.heterograph_index.create_unitgraph_from_coo(
            1 if relation[0] == relation[2] else 2,
            nnodes[relation[0]],
            nnodes[relation[2]],
            edges[relation][0],
            edges[relation][1],
            ['coo', 'csc', 'csr']
        ))

    hgidx = dgl.heterograph_index.create_heterograph_from_relations(
        full_graph._graph.metagraph,
        relations,
        None
    )

    dst_node_ids = [
        dgl.utils.toindex(dst_nodes.get(ntype, []), full_graph._idtype_str).tousertensor(
            ctx=dgl.backend.to_backend_ctx(full_graph._graph.ctx)
        )
        for ntype in full_graph.ntypes
    ]

    dst_node_ids_nd = [dgl.backend.to_dgl_nd(nodes) for nodes in dst_node_ids]

    new_graph_index, src_nodes_ids_nd, induced_edges_nd = dgl._ffi.capi._CAPI_DGLToBlock(
        hgidx, dst_node_ids_nd, True, []
    )

    new_ntypes = (full_graph.ntypes, full_graph.ntypes)
    new_graph = DGLBlock(new_graph_index, new_ntypes, full_graph.etypes)

    src_node_ids = [dgl.backend.from_dgl_nd(src) for src in src_nodes_ids_nd]
    edge_ids = [dgl.backend.from_dgl_nd(eid) for eid in induced_edges_nd]

    node_frames = dgl.utils.extract_node_subframes_for_block(
        full_graph, src_node_ids, dst_node_ids
    )

    edge_frames = dgl.utils.extract_edge_subframes(full_graph, edge_ids)
    dgl.utils.set_new_frames(
        new_graph, node_frames=node_frames, edge_frames=edge_frames
    )

    return new_graph



class PyGSampler(dgl.dataloading.Sampler):
    r"""
    An example DGL sampler implementation that matches PyG/GLT sampler behavior. 
    The following differences need to be addressed: 
    1.  PyG/GLT applies conv_i to edges in layer_i, and all subsequent layers, while DGL only applies conv_i to edges in layer_i. 
        For instance, consider a path a->b->c. At layer 0, 
        DGL updates only node b's embedding with a->b, but 
        PyG/GLT updates both node b and c's embeddings.
        Therefore, if we use h_i(x) to denote the hidden representation of node x at layer i, then the output h_2(c) is: 
            DGL:     h_2(c) = conv_2(h_1(c), h_1(b)) = conv_2(h_0(c), conv_1(h_0(b), h_0(a)))
            PyG/GLT: h_2(c) = conv_2(h_1(c), h_1(b)) = conv_2(conv_1(h_0(c), h_0(b)), conv_1(h_0(b), h_0(a)))
    2.  When creating blocks for layer i-1, DGL not only uses the destination nodes from layer i, 
        but also includes all subsequent i+1 ... n layers' destination nodes as seed nodes.
    More discussions and examples can be found here: https://github.com/alibaba/graphlearn-for-pytorch/issues/79. 
    """
    def __init__(self, fanouts, num_threads=1):
        super().__init__()
        self.fanouts = fanouts
        self.num_threads = num_threads

    def sample(self, g, seed_nodes):
        if self.num_threads != 1:
            old_num_threads = torch.get_num_threads()
            torch.set_num_threads(self.num_threads)
        output_nodes = seed_nodes
        subgs = []
        previous_edges = {}
        previous_seed_nodes = seed_nodes
        input_nodes = seed_nodes
        
        target_labels = g.label[seed_nodes['paper']]

        device = None
        for key in seed_nodes:
            device = seed_nodes[key].device

        n_total_nodes = {key: g.num_nodes(ntype=key) for key in g.ntypes}

        not_sampled = {
            ntype: torch.ones([nnodes], dtype=torch.bool, device=device) for ntype, nnodes in n_total_nodes.items()
        }

        for fanout in reversed(self.fanouts):
            # Sample a fixed number of neighbors of the current seed nodes.
            seed_nodes = {}
            for node_type in previous_seed_nodes:
                seed_nodes[node_type] = previous_seed_nodes[node_type][not_sampled[node_type][previous_seed_nodes[node_type]]]
                not_sampled[node_type][seed_nodes[node_type]] = 0

            sg = g.sample_neighbors(seed_nodes, fanout)

            new_edges = {}
            for etype in sg.canonical_etypes:
                new_edges[etype] = sg.edges(etype=etype)

            # We add all previously accumulated edges to this subgraph
            for etype in previous_edges:
                new_edges[etype] = (
                    torch.cat((new_edges[etype][0], previous_edges[etype][0])),
                    torch.cat((new_edges[etype][1], previous_edges[etype][1]))
                )
            
            previous_edges = new_edges

            # Convert this subgraph to a message flow graph.
            # we need to turn on the include_dst_in_src
            # so that we get compatibility with DGL's OOTB GATConv. 
            # sg = dgl.heterograph(previous_edges, n_total_nodes)
            # sg = dgl.to_block(sg, previous_seed_nodes, include_dst_in_src=True)
            sg = make_block(
                full_graph=g, 
                edges=new_edges,
                nnodes=n_total_nodes,
                dst_nodes=previous_seed_nodes
            )

            # for this layers seed nodes - 
            # they will be our next layers' destination nodes
            # so we add them to the collection of previous seed nodes. 
            previous_seed_nodes = sg.srcdata[dgl.NID]

            # we insert the block to our list of blocks
            subgs.insert(0, sg)
            input_nodes = seed_nodes

        # before we return: assigns the graph label here
        subgs[-1].label = target_labels

        if self.num_threads != 1:
            torch.set_num_threads(old_num_threads)
        return input_nodes, output_nodes, subgs
