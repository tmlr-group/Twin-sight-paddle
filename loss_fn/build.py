import paddle.nn as nn



"""
    args.loss_fn in 
    ["nll_loss", "CrossEntropy"]

"""

def create_loss(args, device=None, **kwargs):
    if "client_index" in kwargs:
        client_index = kwargs["client_index"]
    else:
        client_index = args.client_index

    if args.loss_fn == "CrossEntropy":
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss_fn == "nll_loss":
        loss_fn = nn.NLLLoss()
    else:
        raise NotImplementedError

    return loss_fn















