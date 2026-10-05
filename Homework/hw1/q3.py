###Q3: allreduce###
###implement ring all-reduce with isend/irecv; built-in collectives are not allowed###

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist

def reduce_scatter(chunks, tmp, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration

    for i in range(world-1):
        s = dist.isend(chunks[rank-i], right) # rank 0 sends chunk[0], chunk[-1], chunk[-2], therefore, chunks[rank-i]
        r = dist.irecv(tmp, left) # rank 0 receives rank[-1]
        s.wait()
        r.wait()

        # rank 0 updates chunk[-1] from rank[-1], rank 1 updates chunk[0] from rank[0], chunk[rank - i] update by temmp
        chunks[-1] += temp
    #                                                                   #
    #                                                                   #
    return chunks
        
def all_gather(chunks, tmp, current, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration

    # rank 0 gives chunk[1] to rank[1], then receive chunk[0] from rank 4;
    #        gives chunk[0] to rank[1], then receive chunk[-1] from rank 4

    for i in range(world-1):
        s = dist.isend(chunks[(rank - i + 1) % world], right)
        r = dist.irecv(tmp, left)
        s.wait()
        r.wait()

        chunks[(rank - i + 1) % world] = tmp
    #                                                                   #
    #                                                                   #
    return chunks

def ring_allreduce_(tensor: torch.Tensor, world_size = None, rankid = None):
    """In-place ring all-reduce average using isend/irecv."""
    world = world_size
    if world == 1: return tensor
    rank = rankid
    left, right = (rank - 1) % world, (rank + 1) % world

    ##following steps try to fill blank to the tensor so that final tensor can be divided to 3 chunks evenly
    flat = tensor.contiguous().view(-1)
    n = flat.numel()
    chunk = (n + world - 1) // world
    #                                                                   #
    #                                                                   #
    # your code here: we cannot divide flat into 3 pieces evenly as the
    # flat lengh may not be able to divided exactly by 3....
    #
    #                                                                   #
    #                                                                   #
    #So, fill zeros at the end of flat to generate padded_flat
    pad_config = (0, chunk * world - n)
    padded_flat = torch.nn.functional.pad(flat, pad_config, model="constant", value=0.0) # modify this line and fill correct value into padded_flat
    chunks = [padded_flat[i*chunk:(i+1)*chunk] for i in range(world)]

    #                                                                   #
    #                                                                   #
    # your code here: call reduce_scatter and all_gather
    chunks = reduce_scatter(chunks, tmp, world, rank, left, right)
    chunks = allreduce(chunks, tmp, rank, world, rank, left, right)
    #
    #                                                                   #
    #                                                                   #
    #we provide the reduce_scatter and all_gather func prototype for you
    # You may adjust the function signature (input structure) of `reduce_scatter` and `all_gather` if needed.
    
    # stitch & unpad  
    flat /= world
    tensor.view(-1).copy_(flat[:n])
    return
