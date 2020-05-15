import torch
from torch.autograd.functional import _as_tuple, _grad_preprocess, _check_requires_grad, _validate_v, _autograd_grad, _fill_in_zeros, _grad_postprocess, _tuple_postprocess


def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])
        
def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)
        
def make_functional(mod):
    orig_params = tuple(mod.parameters())
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

def fw_linearize(func, inputs, create_graph=False, strict=False):
    is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jvp")
    inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

    outputs = func(*inputs)
    is_outputs_tuple, outputs = _as_tuple(outputs, "outputs of the user-provided function", "jvp")
    _check_requires_grad(outputs, "outputs", strict=strict)
    # The backward is linear so the value of grad_outputs is not important as
    # it won't appear in the double backward graph. We only need to ensure that
    # it does not contain inf or nan.
    grad_outputs = tuple(torch.zeros_like(out, requires_grad=True) for out in outputs)

    grad_inputs = _autograd_grad(outputs, inputs, grad_outputs, create_graph=True)
    _check_requires_grad(grad_inputs, "grad_inputs", strict=strict)

    def lin_fn(v, retain_graph=True):
        if v is not None:
            _, v = _as_tuple(v, "v", "jvp")
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, inputs, is_inputs_tuple)
        else:
            if len(inputs) != 1 or inputs[0].nelement() != 1:
                raise RuntimeError("The vector v can only be None if the input to "
                                   "the user-provided function is a single Tensor "
                                   "with a single element.")

        grad_res = _autograd_grad(grad_inputs, grad_outputs, v, create_graph=create_graph, retain_graph=retain_graph)

        jvp = _fill_in_zeros(grad_res, outputs, strict, create_graph, "back_trick")

        # Cleanup objects and return them to the user
        jvp = _grad_postprocess(jvp, create_graph)

        return _tuple_postprocess(jvp, is_outputs_tuple)
    return lin_fn, outputs