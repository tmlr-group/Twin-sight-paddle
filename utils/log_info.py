

def log_info(type:str, name:str, info, step=None, record_tool='wandb'):
    '''
    type: the info type mainly include: image, scalar (tensorboard may include hist, scalars)
    name: replace the info name displayed in wandb or tensorboard
    info: info to record
    '''
    if record_tool == 'wandb':
        import wandb
    if type == 'image':
        if record_tool == 'tensorboard' and False:
            writer.add_image(name, info, step)
        if record_tool == 'wandb' and False:
            wandb.log({name: wandb.Image(info)})

    if type == 'scalar' and False:
        if record_tool == 'tensorboard' and False:
            writer.add_scalar(name, info, step)
        if record_tool == 'wandb' and False:
            wandb.log({name:info})
    if type == 'histogram' and False:
        writer.add_histogram(name, info, step)