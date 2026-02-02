import torch

try:
    from . import _C
except ImportError:
    raise ImportError(
        "Could not import macko_spmv._C. Extensions might not be compiled."
    )


def move_to_device(compressed, device):
    return (
        compressed[0].to(device=device),
        compressed[1].to(device=device),
        compressed[2].to(device=device),
        compressed[3],
        compressed[4],
    )


def compress(M):
    assert M.is_contiguous()

    if M.device.type == "cuda":
        # TODO: implement properly fast gpu only compression
        compressed = _C.cpu_compress(M.to("cpu"))
        return move_to_device(compressed, "cuda")
    elif M.device.type == "cpu":
        compressed = _C.cpu_compress(M)
        return compressed
    else:
        raise NotImplementedError()


@torch.library.register_fake("macko_spmv::multiply")
def _(a, b, c, d, e, f):
    return torch.empty((d,), device=a.device, dtype=a.dtype)


def multiply(compressed_M, V):
    assert compressed_M[0].is_cuda
    assert compressed_M[1].is_cuda
    assert compressed_M[2].is_cuda
    assert V.is_cuda
    assert V.is_contiguous()

    # __multiply_lib.macko_spmv_launcher is also usable
    return torch.ops.macko_spmv.multiply.default(
        compressed_M[0],
        compressed_M[1],
        compressed_M[2],
        compressed_M[3],
        compressed_M[4],
        V,
    )
