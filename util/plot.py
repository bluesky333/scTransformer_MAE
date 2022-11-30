import pandas as pd
import matplotlib.pyplot as plt

def plot_model_output_histogram(gnd, model_out, mask):
    model_out = model_out.squeeze(-1)
    model_out = model_out.detach().cpu().numpy()
    gnd = gnd.detach().cpu().numpy()
    mask = mask.cpu()
    gnd_flattened = []
    model_output_flattened = []
    
    for cell_idx in range(gnd.shape[0]):
        gnd_flattened += list(gnd[cell_idx][mask[cell_idx].bool()])  # x
        model_output_flattened += list(model_out[cell_idx][mask[cell_idx].bool()])  # y on scatterplot
    
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.hist(gnd_flattened,bins=20)
    ax1.set_xlabel('Actual Value')
    ax2.hist(model_output_flattened,bins=20)
    ax2.set_xlabel('Model Output Value')
    return fig

def plot_scatterplot(gnd, model_out, labels, mask):  # ToDO: docstring
    model_out = model_out.squeeze(-1)
    model_out = model_out.detach().cpu().numpy()
    gnd = gnd.detach().cpu().numpy()
    mask = mask.cpu()
    gnd_vals_list = []
    pred_vals_list = []
    labels_list = []
    for cell_idx in range(gnd.shape[0]):
        gnd_vals_list += list(gnd[cell_idx][mask[cell_idx].bool()])  # x
        pred_vals_list += list(model_out[cell_idx][mask[cell_idx].bool()])  # y on scatterplot
        labels_list += [labels[cell_idx] for _ in range(int(mask[cell_idx].sum().item()))]

    # Create Pandas DataFrame
    plot_df = pd.DataFrame({
        "Ground Truth Expression": gnd_vals_list,
        "Predicted Expression": pred_vals_list,
        "Cell Class": labels_list
    })
    fig = plt.figure()
    plt.scatter(plot_df['Ground Truth Expression'], plot_df['Predicted Expression'], c = plot_df['Cell Class'], figure=fig)
    
    return fig