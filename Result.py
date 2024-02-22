import corner
import matplotlib.pyplot as plt
import numpy as np

class Result():
    '''
    result: result from sampler
    result_type: 'dynesty'
    result_model: model name
    '''
    def __init__(self,result,result_type:str,result_model:str,suf:str,param_names=None):
        
        self.result = result
        self.result_type = result_type
        self.result_model = result_model
        self.suf = suf
        self.param_names = param_names
        self.samples = self.get_samples()
        self.best_params = np.median(self.samples,axis=0)

    def plot_corner(self,is_save = True):
        
        samples_plot,params_plot = self.get_plotsamples()
        truths_plot = np.median(samples_plot,axis = 0)
        
        fig = corner.corner(samples_plot,labels=params_plot,quantiles=[0.16, 0.84],show_titles=True,
                            color = 'C0',plot_datapoints = False,plot_contours = True,fill_contours = True,label_kwargs={'labelpad':0})#,truths=truths_plot)

        axes = np.array(fig.axes).reshape((len(self.param_names),len(self.param_names)))


        for i,ax in enumerate(axes.diagonal()):
            ax.axvline(truths_plot[i],linestyle = 'solid')

        if is_save:
            plt.savefig(f'./results/{self.result_type}_{self.result_model}_{self.suf}/corner_plot.pdf')
        
        return fig