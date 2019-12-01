import matplotlib.pyplot as plt

def plot(data_x, title, filename, legend=['Agent1', 'Agent2'], xlabel='Epochs', ylabel='Cumulative data_x'):
    cmap = ['red', 'blue', 'green', 'black', 'gray']

    # Plot L of class 0
    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
    
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Provide tick lines across the plot to help your viewers trace along the axis ticks. 
    plt.grid(True, which='major', axis='y', ls='--', lw=0.5, c='k', alpha=0.3)
    
#    plt.tick_params(axis='both', which='both', bottom=True, top=True,
#                   labelbottom=True, left=True, right=True, labelleft=True)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    runs = legend

    plt.title(title, fontsize=16)
    for i in range(2):
        plt.plot(range(1,len(data_x[0])+1), data_x[i], lw=2.5, color = cmap[i], label=runs[i])

    plt.legend(ncol=2)
    
    fig.savefig(filename+'.png', figsize=(12,8))
    fig.clear()
