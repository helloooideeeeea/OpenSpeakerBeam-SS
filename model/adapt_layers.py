import torch
import torch.nn as nn


def into_tuple(x):
    """
    Transforms tensor/list/tuple into tuple.
    """
    if isinstance(x, list):
        return tuple(x)
    elif isinstance(x, torch.Tensor):
        return (x,)
    elif isinstance(x, tuple):
        return x
    else:
        raise ValueError('x should be tensor, list of tuple')

def into_orig_type(x, orig_type):
    """
    Inverts into_tuple function.
    """
    if orig_type is tuple:
        return x
    if orig_type is list:
        return list(x)
    if orig_type is torch.Tensor:
        return x[0]
    else:
        assert False


class MulAddAdaptLayer(nn.Module):
    def __init__(self, indim=256, enrolldim=256, ninputs=1, do_addition=False):
        super().__init__()
        self.ninputs = ninputs
        self.do_addition = do_addition

        assert ((do_addition and enrolldim == 2*indim) or \
                (not do_addition and enrolldim == indim))

    def forward(self, main, enroll):
        """
        Arguments:
            main: tensor or tuple or list
                  activations in the main neural network, which are adapted
                  tuple/list may be useful when we want to apply the adaptation
                    to both normal and skip connection at once
            enroll: tensor or tuple or list
                    embedding extracted from enrollment
                    tuple/list may be useful when we want to apply the adaptation
                      to both normal and skip connection at once
        """
        assert type(main) == type(enroll)
        orig_type = type(main)
        main, enroll = into_tuple(main), into_tuple(enroll)
        assert len(main) == len(enroll) == self.ninputs

        out = []
        for main0, enroll0 in zip(main, enroll):
            if self.do_addition:
                enroll0_mul, enroll0_add = torch.chunk(enroll0, 2, dim=1)
                out.append(enroll0_mul[...,None] * main0 + enroll0_add[...,None])
            else:
                out.append(enroll0[...,None] * main0)
        return into_orig_type(tuple(out), orig_type)

