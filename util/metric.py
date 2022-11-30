from scipy.stats import pearsonr

def calculate_r_squared(pred, images):
    """
    Helper function to calculate R-squared between predicted pixel values and actual
    masked pixel values

    Args:
        pred: Tensor of model predictions, shape torch.Size([64 batch size, 784 pixels, 1])

    """
    pred = pred.squeeze(-1)
    pred = pred.flatten().detach().cpu().numpy()
    images = images.flatten().detach().cpu().numpy()

    pearson_r, p_val = pearsonr(x=pred, y=images)
    return pearson_r ** 2

def calculate_r_squared_masked(pred, images, mask):
    """
    Helper function to calculate R-squared between predicted pixel values and actual
    masked pixel values

    Args:
        pred: Tensor of model predictions, shape torch.Size([64 batch size, 784 pixels, 1])

    """
    pred = pred.squeeze(-1)
    pred = pred.detach().cpu().numpy()
    images = images.detach().cpu().numpy()
    mask = mask.cpu()
    
    images_list = []
    pred_vals_list = []
    for cell_idx in range(images.shape[0]):
        images_list += list(images[cell_idx][mask[cell_idx].bool()]) # masked ground truth 
        pred_vals_list += list(pred[cell_idx][mask[cell_idx].bool()]) # masked prediction 

    pearson_r, p_val = pearsonr(x=pred_vals_list, y=images_list)
    return pearson_r ** 2