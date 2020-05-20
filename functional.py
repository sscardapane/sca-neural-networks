import torch

# Code here is mostly taken from this discussion:
# https://discuss.pytorch.org/t/combining-functional-jvp-with-a-nn-module/81215/7

def del_attr(obj, names):
    # Recursively deletes all attributes in names from obj
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])
        
def set_attr(obj, names, val):
    # Recursively set all attributes (names, val) in obj
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)
        
def make_functional(mod):
    orig_params = list(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

def functional_mod_fw(xb, mod, names, *params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)
    return mod(xb)

#def _grad_preprocess(inputs):
#    res = tuple([inp.view_as(inp) for inp in inputs])
#    return res

def fw_linearize(func, inputs):
    #inputs = _grad_preprocess(inputs)
    outputs = func(*inputs)

    # The backward is linear so the value of grad_outputs is not important as
    # it won't appear in the double backward graph. We only need to ensure that
    # it does not contain inf or nan.
    grad_outputs = torch.zeros_like(outputs, requires_grad=True)

    grad_inputs = torch.autograd.grad(outputs, inputs, grad_outputs, allow_unused=True,
                                   create_graph=True, retain_graph=None)

    def lin_fn(v, retain_graph=True):
        #v = _grad_preprocess(v)
        jvp = torch.autograd.grad(grad_inputs, grad_outputs, v, allow_unused=True,
                                   create_graph=True, retain_graph=True)
        return jvp
    return lin_fn, outputs