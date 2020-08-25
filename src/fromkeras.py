from keras.models import load_model
from model import SketchKeras
import torch

if __name__ == "__main__":

    model = load_model("weights/mod.h5")

    conv_weight = dict()
    batch_weight = dict()
    conv_idx = 0
    batch_idx = 0
    for layer in model.layers:
        weights = layer.get_weights()

        layer_name = layer.name

        if layer_name.startswith("conv2d"):
            w, b = weights
            print(layer_name, w.shape, b.shape, w.dtype, b.dtype)
            conv_weight[f"conv_{conv_idx}_weight"] = w
            conv_weight[f"conv_{conv_idx}_bias"] = b
            conv_idx += 1

        if layer_name.startswith("batch"):
            print(
                layer_name,
                weights[0].shape,
                weights[1].shape,
                weights[2].shape,
                weights[3].shape,
            )
            batch_weight[f"batch_{batch_idx}_weight"] = weights[0]
            batch_weight[f"batch_{batch_idx}_bias"] = weights[1]
            batch_weight[f"batch_{batch_idx}_running_mean"] = weights[2]
            batch_weight[f"batch_{batch_idx}_running_var"] = weights[3]
            batch_idx += 1

    print("-" * 20)

    conv_idx = 0
    batch_idx = 0
    torchmodel = SketchKeras()
    for name, module in torchmodel.named_children():

        for submodule in module.modules():
            submodule_name = submodule._get_name()

            if submodule_name.startswith("Conv2d"):
                w = conv_weight[f"conv_{conv_idx}_weight"].transpose(3, 2, 0, 1)
                b = conv_weight[f"conv_{conv_idx}_bias"]
                submodule.state_dict()["weight"].copy_(torch.tensor(w))
                submodule.state_dict()["bias"].copy_(torch.tensor(b))

                print(
                    submodule_name,
                    submodule.state_dict()["weight"].shape,
                    submodule.state_dict()["bias"].shape,
                )
                conv_idx += 1

            if submodule_name.startswith("BatchNorm2d"):
                a = batch_weight[f"batch_{batch_idx}_weight"]
                b = batch_weight[f"batch_{batch_idx}_bias"]
                c = batch_weight[f"batch_{batch_idx}_running_mean"]
                d = batch_weight[f"batch_{batch_idx}_running_var"]
                submodule.state_dict()["weight"].copy_(torch.tensor(a))
                submodule.state_dict()["bias"].copy_(torch.tensor(b))
                submodule.state_dict()["running_mean"].copy_(torch.tensor(c))
                submodule.state_dict()["running_var"].copy_(torch.tensor(d))

                print(
                    submodule_name,
                    submodule.state_dict()["weight"].shape,
                    submodule.state_dict()["bias"].shape,
                    submodule.state_dict()["running_mean"].shape,
                    submodule.state_dict()["running_var"].shape,
                    submodule.state_dict()["num_batches_tracked"].shape,
                )

                batch_idx += 1

    torch.save(torchmodel.state_dict(), "weights/model.pth")
