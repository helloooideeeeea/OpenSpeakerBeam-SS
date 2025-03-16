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


def main():
    # テスト用の簡単なテンソルを作成
    # main: バッチサイズ 2, チャンネル数 indim=2, 時間軸 T=3
    main_tensor = torch.tensor([[[1.0, 2.0, 3.0],
                                 [4.0, 5.0, 6.0]],
                                [[7.0, 8.0, 9.0],
                                 [10.0, 11.0, 12.0]]])
    print("main_tensor shape:", main_tensor.shape)
    # MulAddAdaptLayer は do_addition=False の場合、enroll のチャネル数は indim と同じ (ここでは2)
    layer = MulAddAdaptLayer(indim=2, enrolldim=2, ninputs=1, do_addition=False)

    # ケース1: enroll のバッチサイズが 1 の場合
    enroll_case1 = torch.tensor([[0.5, 2.0]])  # shape (1, 2)
    print("\nTest case 1: enroll batch size 1")
    print("enroll_case1 shape:", enroll_case1.shape)
    output1 = layer(main_tensor, enroll_case1)
    print("Output shape:", output1.shape)
    print("Output:")
    print(output1)
    # ※ enroll_case1 の値が自動的にバッチ次元でブロードキャストされ、
    #     全ての main サンプルに対して同じエンベディングが適用される

    # ケース2: enroll のバッチサイズが 2 の場合
    enroll_case2 = torch.tensor([[0.5, 2.0],
                                 [1.0, 3.0]])  # shape (2, 2)
    print("\nTest case 2: enroll batch size 2")
    print("enroll_case2 shape:", enroll_case2.shape)
    output2 = layer(main_tensor, enroll_case2)
    print("Output shape:", output2.shape)
    print("Output:")
    print(output2)
    # ※ この場合、各 main サンプルに対して対応する enroll のエンベディングが掛け合わされる


if __name__ == "__main__":
    main()