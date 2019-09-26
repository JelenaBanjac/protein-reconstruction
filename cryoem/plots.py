import matplotlib.pyplot as plt

fg_color = 'white'
bg_color = 'black'

def _plot(image, title, ax, colorbar=False, mean=0, var=0):
    im = ax.imshow(image)
  
    # set title plus title color
    ax.set_title(title, color=fg_color)

    # set figure facecolor
    ax.patch.set_facecolor(bg_color)

    # set tick and ticklabel color
    im.axes.tick_params(color=fg_color, labelcolor=fg_color)

    # set imshow outline
    for spine in im.axes.spines.values():
        spine.set_edgecolor(fg_color)    

    if colorbar:
        cb = plt.colorbar(im)
        # set colorbar label plus label color
        cb.set_label('Closeness', color=fg_color)

        # set colorbar tick color
        cb.ax.yaxis.set_tick_params(color=fg_color)

        # set colorbar edgecolor 
        cb.outline.set_edgecolor(fg_color)

        # set colorbar ticklabels
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=fg_color)
        
def plot_projection(image, title, mean=0, var=0):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    
    _plot(image, title, ax, colorbar=True)

    fig.patch.set_facecolor(bg_color) 
    plt.tight_layout()
    
    plt.show()
    
def plot_projections(images, titles, nrows=2, ncols=5):
    fig, axes = plt.subplots(nrows, ncols, figsize=(25, 10))
    
    cr = [(i, j) for i in range(nrows) for j in range(ncols)]
    
    for image, title, (i, j) in zip(images, titles, cr):
        
        _plot(image, title, axes[i][j] if nrows>1 else axes[j], colorbar=False)

    fig.patch.set_facecolor(bg_color)    
    plt.tight_layout()

    plt.show()