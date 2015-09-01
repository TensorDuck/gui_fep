"""
This is a work in progress:

I want to make a GUI plot of a free-energy landscape that can:

essential:

- plot free energy
- Mark a set of bins
- render a box around them in the GUI
- Output a .ndx file containing groups for the set of bins that was selected
- Be able to output multiple .ndx files
- Be able to dynamically grow a .ndx file

Would be nice:
- Dynamically adjust bin-size
- Be able to look back and render multiple boxes of selected points


Written by Justin Chen while working with Cecilia Clementi as a Physics PhD student at Rice University
"""

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from scipy.sparse import spdiags
import scipy.stats as stats

kb = 6.022141*1.380650/(4.184*1000.0)

###Classes for working with GUI

##class for holding universal types of data shared between different methods
class UniversalDataHolder:
    def __init__(self):
        self.group_number = 0
        
##class for all selectin tools

class selection_tool:

    def __init__(self):
        pass
    def connect(self):
        self.cid_key_save = self.box.figure.canvas.mpl_connect("key_press_event", self.key_save)
    def disconnect(self):
        self.box.figure.canvas.mpl_disconnect(self.cid_key_save)
        
    def key_save(self, event):
        if event.key == "c":
            self.save_frames()
    
    def save_frames(self):
        pass     

##Begin ModeSelect
class ModeSelect:
    def __init__(self, axes, dc1, dc2, slices, xedges, yedges, append=False):
        self.dc1 = dc1
        self.dc2 = dc2
        self.slices = slices
        self.udata = UniversalDataHolder()
        self.x_count = 0
        self.xedges = xedges
        self.yedges = yedges
        
        #deal with the files list
        if append:
            #open files for append
            self.file_info = open("frames-info.txt", "a", 0)
            self.file = open("frames.ndx", "a", 0)
        else:
            #open files for writing
            self.file_info = open("frames-info.txt", "w", 0)
            self.file = open("frames.ndx", "w", 0)
        
        #initialize the selection methods and their related attributes
        self.axes = axes
        self.mode_selectbox = SelectBox(self.axes, self.dc1, self.dc2, self.udata, self.file, self.file_info)
        self.mode_selectbin = SelectBin(self.axes, self.dc1, self.dc2, self.slices, self.xedges, self.yedges, self.udata, self.file, self.file_info)
        self.select_mode = self.mode_selectbox
        self.select_mode.connect()
        
        #start up so that key presses get tied to here key_selection() method below
        self.cid_key_choose = self.axes.figure.canvas.mpl_connect("key_press_event", self.key_selection)

       
    def key_selection(self, event):
        if event.key == "x":
            self.select_mode.disconnect()
            self.x_count += 1
            if self.x_count == 3:
                self.axes.figure.canvas.mpl_disconnect(self.cid_key_choose)
                self.file.close()
                self.file_info.close()
                print "Thank you for using the gui_fep package. Have a nice day!"
        else:
            self.x_count = 0
            if event.key == "1":
                self.select_mode.disconnect()
                self.select_mode = self.mode_selectbox
                self.select_mode.connect()
            if event.key == "2":
                self.select_mode.disconnect()
                self.select_mode = self.mode_selectbin
                self.select_mode.connect()
            
#Begin Simple Box
class SimpleBox(selection_tool):
    """class SimpleBox acts as a class of methods for managing a simple selection box GUI"""
    def __init__(self, axes):
        self.box = axes.add_patch(patches.Rectangle((0,0),0,0, fill=False, edgecolor="k", linewidth=2))
        self.press = False
        self.busy = False
        self.name = "SimpleBox"
        print "Got to init"
    
    def connect(self):
        selection_tool.connect(self) #connect the supper class events
        print "Connecting %s" % self.name
        self.cid_press = self.box.figure.canvas.mpl_connect("button_press_event", self.on_press)        
        self.cid_move = self.box.figure.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.cid_release = self.box.figure.canvas.mpl_connect("button_release_event", self.on_release)
        
        #make sure it's visible
        self.box.set_visible(True)
        
        
        
    def disconnect(self):
        print "Disconnecting %s" % self.name
        selection_tool.disconnect(self) #disconnect the super class events
        self.box.set_visible(False) #make invisible now
        self.box.figure.canvas.draw()
        self.box.figure.canvas.mpl_disconnect(self.cid_press)        
        self.box.figure.canvas.mpl_disconnect(self.cid_move)
        self.box.figure.canvas.mpl_disconnect(self.cid_release)   
        
    def on_press(self, event):
        print "pressing"
        if self.busy:
            return
        self.box.set_x(event.xdata)
        self.box.set_y(event.ydata)
        self.box.set_width(0)
        self.box.set_height(0)
        self.press = True
        self.box.figure.canvas.draw()
        
    def on_move(self, event):
        if self.press:
            self.box.set_width(event.xdata - self.box.get_x())
            self.box.set_height(event.ydata - self.box.get_y())
            self.box.figure.canvas.draw()
            
    def on_release(self, event):
        if self.busy:
            return
        print "releasing"
        self.press = False
        self.box.figure.canvas.draw()
        print "box vertices are at: "
        print self.get_vertices()
        print "The bounds are at: "
        print self.get_bounds()
        
    def get_vertices(self):
        xlocation = self.box.get_x()
        ylocation = self.box.get_y()
        height = self.box.get_height()
        width = self.box.get_width()
        
        xs = np.array([xlocation, xlocation+width])
        ys = np.array([ylocation, ylocation+height])
        
        vertices = np.array([[xs.min(), ys.min()], [xs.min(), ys.max()], [xs.max(), ys.max()], [xs.max(), ys.min()]])
        
        return vertices
        
    def get_bounds(self):
        xlocation = self.box.get_x()
        ylocation = self.box.get_y()
        height = self.box.get_height()
        width = self.box.get_width()
        
        xs = np.array([xlocation, xlocation+width])
        ys = np.array([ylocation, ylocation+height])
        
        return np.array([xs.min(), xs.max(), ys.min(), ys.max()])

#Begin BinSelect
class SelectBin(SimpleBox):
    def __init__(self, axes, dc1, dc2, slices, xedges, yedges, udata, file_save, file_info):
        SimpleBox.__init__(self, axes)
        self.dc1 = dc1
        self.dc2 = dc2
        self.slices = slices
        self.xedges = xedges
        self.yedges = yedges
        
        
        self.udata = udata
        self.file = file_save
        self.file_info =file_info
        
        self.name = "SelectBin"
        print "Initialized SelectBins"
    
    def on_release(self, event):
        if self.busy:
            return
        self.busy=True
        self.press = False
        
        
        bounds = self.get_bounds()
        possible_small_x = self.xedges[self.xedges <= bounds[0]]
        possible_big_x = self.xedges[self.xedges >= bounds[1]]
        possible_small_y = self.yedges[self.yedges <= bounds[2]]
        possible_big_y = self.yedges[self.yedges>= bounds[3]]
        
        small_x = np.max(possible_small_x)
        big_x = np.min(possible_big_x)
        small_y = np.max(possible_small_y)
        big_y = np.min(possible_big_y)
        
        self.box.set_x(small_x)
        self.box.set_y(small_y)
        self.box.set_width(big_x - small_x)
        self.box.set_height(big_y - small_y)
        self.box.figure.canvas.draw()
        
        print "The bounds are at: "
        print self.get_bounds()
        
        self.busy=False
        
    def save_frames(self):
        print "saving frames"
        self.busy = True
        bounds = self.get_bounds()
        self.file_info.write("[Group %d]\nBounds are from: \n%f < dc1 < %f \n%f < dc2 < %f\n" % (self.udata.group_number, bounds[0],bounds[1],bounds[2],bounds[3]))
        self.file.write("[group %d]\n"%self.udata.group_number)
        for i in range(np.shape(self.dc1)[0]):
            if self.dc1[i] >= bounds[0] and self.dc1[i] <= bounds[1] and self.dc2[i] >= bounds[2] and self.dc2[i] <= bounds[3]:
                self.file.write("%d\n" % (i+1)) 
        self.udata.group_number += 1
        self.busy = False
        print "done saving frames"
        
        

##Begin SelectBox
class SelectBox(SimpleBox):
    def __init__(self, axes, dc1, dc2, udata, file_save, file_info):
        SimpleBox.__init__(self, axes)
        self.dc1 = dc1
        self.dc2 = dc2
        self.udata = udata
        self.file = file_save
        self.file_info =file_info
        self.name = "SelectBox"
        print "Initialized SelectBins"
        
    def save_frames(self):
        print "saving frames"
        self.busy = True
        bounds = self.get_bounds()
        self.file_info.write("[Group %d]\nBounds are from: \n%f < dc1 < %f \n%f < dc2 < %f\n" % (self.udata.group_number, bounds[0],bounds[1],bounds[2],bounds[3]))
        self.file.write("[group %d]\n"%self.udata.group_number)
        for i in range(np.shape(self.dc1)[0]):
            if self.dc1[i] >= bounds[0] and self.dc1[i] <= bounds[1] and self.dc2[i] >= bounds[2] and self.dc2[i] <= bounds[3]:
                self.file.write("%d\n" % (i+1)) 
        self.udata.group_number += 1
        self.busy = False
        print "done saving frames"
        
##Begin SelectMultiBin
class SelectMultiBin(SimpleBox):
    def __init__(self, axes, dc1, dc2, slices, xedges, yedges, udata, file_save, file_info):
        SimpleBox.__init__(self, axes)
        self.dc1 = dc1
        self.dc2 = dc2
        self.slices = slices
        
        self.xedges = xedges
        self.yedges = yedges
        
        self.num_xbins = np.shape(xedges)[0] - 1
        self.num_xbins_all = self.num_xbins + 2
        self.num_ybins = np.shape(yedges)[0] - 1
        #assign an extra edge to the beginning and end of the _all lists. 
        #This is because SciPy hashes all values outside of the range into extra bins before and after the range
        #This happens in both directions and consequently adds 2 to each dimension of the matrix 
        self.xedges_all = np.array(xedges)
        self.yedges_all = np.array(yedges)
        
        #make an array keeping track of which bins have been selected.
        self.selected_bins = np.zeros(self.num_xbins, self.num_ybins)
        self.line_lists = []
        for i in range(1):
            line_lists.append(lines.Line2D([0,0], [0,0], color='k')
        
        
        self.udata = udata
        self.file = file_save
        self.file_info =file_info
        
        self.name = "SelectBin"
        print "Initialized SelectMultiBins"
    
    def save_frames(self):
        print "saving frames"
        self.busy = True
        self.file_info.write("[Group %d]\nBounds are: \n%f < dc1 < %f \n%f < dc2 < %f\n" % (self.udata.group_number, np.min(self.xedges), np.max(self.xedges), np.min(self.yedges), np.max(self.yedges)))
        real_string = "\n"
        selected_frames = []
        for i in xrange(self.num_xbins):
            temp_string = ""
            for j in xrange(self.num_ybins):
                temp_string += "%2d" % self.selected_bins[i,j] ##collect all the values of 1 and 0 for this row
                selected_frames.append((i+1)*(self.num_xbins_all) + j + 1)
            real_string = temp_string + "\n" real_string ##add them into the real string
        
        for idx, value in slices: ##write the ndx file
            if value in selected_frames:
                self.file.write("%d\n" % (idx+1)) 
                
        self.file_info.write(real_string) #save a grid in the formatted way
        self.udata.group_number += 1
        self.busy = False
        print "done saving frames"
###END CLASSES ^^^






###################################################################################

def smooth_iter(arrayin, N):
    for i in range(N):
        arrayin=smooth2a(arrayin, 1, 1)
    return arrayin
    
def smooth2a(arrayin, nr, nc):
    #shamelessly stolen from Jordane's densplot
    
    # Building matrices that will compute running sums.  The left-matrix, eL,
    # smooths along the rows.  The right-matrix, eR, smooths along the
    # columns.  You end up replacing element "i" by the mean of a (2*Nr+1)-by- 
    # (2*Nc+1) rectangle centered on element "i".
    
    a = arrayin.mask

    row = arrayin.shape[0]
    col = arrayin.shape[1]

    el = spdiags(np.ones((2*nr+1, row)),range(-nr,nr+1), row, row).todense()
    er = spdiags(np.ones((2*nc+1, col)), range(-nc,nc+1), col, col).todense()

    # Setting all "nan" elements of "arrayin" to zero so that these will not
    # affect the summation.  (If this isn't done, any sum that includes a nan
    # will also become nan.)
    maskedarrayin = np.ma.masked_where(a, arrayin)
    arrayin[a] = maskedarrayin.max()

    # For each element, we have to count how many non-nan elements went into
    # the sums.  This is so we can divide by that number to get a mean.  We use
    # the same matrices to do this (ie, "el" and "er").

    nrmlize = el.dot((~a).dot(er))
    nrmlize[a] = None

    # Actually taking the mean.

    arrayout = el.dot(arrayin.dot(er))
    arrayout = arrayout/nrmlize

    return arrayout

def get_test_case():
    data = np.loadtxt("short_traj.ev")
    dcA = data[:,1]
    dcB = data[:,2]
    
    print "min and max of A is:"
    print np.min(dcA)
    print np.max(dcA)
    print "min and max of B is:"
    print np.min(dcB)
    print np.max(dcB)
    
    z,x,y = np.histogram2d(dcA, dcB, bins=[50,50], normed=True)

    min_prob = np.min(z)
    
    zmasked = np.ma.masked_where(z==0, z)
    
    print zmasked.min()
    
    #z = np.ma.masked_where(hist==0, hist)
    z = np.log(z)    
    z *= -1
    z *= kb
    z *= 170

    #z = smooth_iter(z, 2)
    
    fe = np.ma.masked_where(z==float("inf"), z)
    
    fe = fe - fe.min() 
    
    fe_smoothed = smooth_iter(fe, 1)
    #fe_smoothed = smooth2a(fe, 3, 3)
    #fe_smoothed = fe
    return x,y, fe_smoothed, dcA, dcB

#def onclick(event):
    print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata)
    
def get_data_array(load, skip, column):
    data = np.loadtxt(load, skiprows=skip)
    if data.ndim == 1:
        if column == 0:
            return data
        else:
            raise IOError("Column %d does not exist in file %s. File %s is a 1-D array" % (column, load, load))
    else:
        if column >= np.shape(data)[1]:
            raise IOError("Column %d does not exist in file %s. Check to make sure you started counting columns with 0 and not 1" % (column, load))   
        else:   
            return data[:,column]
        
def get_DCs(args):
    if args.sub_type == "same":
        data = np.loadtxt(args.dc_file, skiprows=args.skiprows)
        dcA = data[:,args.dc_use[0]]
        dcB = data[:,args.dc_use[1]]     
    elif args.sub_type == "diff":
        dcA = get_data_array(args.files[0], args.skiprows[0], args.use_columns[0])
        dcB = get_data_array(args.files[1], args.skiprows[1], args.use_columns[1])
    else:
        raise IOError("could not determine data type. Please specify either 'same' or 'diff'") 
    
    return dcA, dcB
def histogram_DCs(dcA, dcB, bin_size, ran_size, temperature, smooth_param):
    if ran_size == None:
        z, x, y, slices = stats.binned_statistic_2d(dcA, dcB, np.ones(np.shape(dcA)[0]), bins=bin_size, statistic='sum')
    else:
        z, x, y, slices = stats.binned_statistic_2d(dcA, dcB, np.ones(np.shape(dcA)[0]), bins=bin_size, range=np.reshape(ran_size,(2,2)), statistic='sum')
    #z,x,y = np.histogram2d(dcA, dcB, bins=[bin_size,bin_size], normed=True)

    min_prob = np.min(z)
    
    zmasked = np.ma.masked_where(z==0, z)

    z = np.log(z)    
    z *= (-1.0) 
    z *= kb 
    z *= temperature
    
    fe = np.ma.masked_where(z==float("inf"), z)
    
    fe = fe - fe.min() 
    
    if smooth_param == 0:
        fe_smoothed = fe
    else:
        fe_smoothed = smooth2a(fe, smooth_param, smooth_param)
    
    return x,y, fe_smoothed, slices
    
def get_args():
    par = argparse.ArgumentParser(description="parent set of parameters", add_help=False)
    par.add_argument("--save_dir", default=os.getcwd(), type=str, help="directory for saving the plots")
    par.add_argument("--save_name", type=str, help="Specify the name of the file to save the .png file of the free energy to")
    par.add_argument("--interactive", default=False, action="store_true", help="specify the interactive mode for selecting data points")
    par.add_argument("--append", default=False, action="store_true", help="use this flag to append to all .ndx files of the same name")
    par.add_argument("--bins", type=int, nargs=2, default=[50,50], help="Specify number of bins in each direction of DC1 and DC2")
    par.add_argument("--range", type=float, default=None, nargs=4, help="Range for binning the data")
    par.add_argument("--temp", type=float, nargs="+", default=300, help="specify the temperature for the data, can be an array")
    par.add_argument("--smooth", type=int, default=1, help="specify the amount of smoothing, higher=more smooth")

    
    
    parser = argparse.ArgumentParser(description="Options for gui_fep script. Use either diff for two different files or same for the same file. Whenever numbers of columns or rows need to be specified, use the Python numbering scheme for those, meaning starting with 0")
    sub = parser.add_subparsers(dest="sub_type")
    
    #diff_sub for analyinzg two files containing the DCs in two separate columns
    diff_sub = sub.add_parser("diff", parents=[par], help="For using data from two different files")
    diff_sub.add_argument("--files", type=str, nargs=2, help="Specify which two files to get data from for plotting")
    diff_sub.add_argument("--skiprows", nargs=2, default=[0,0], type=int, help="Specify the number of rows to skip in each file individually, takes two arguments")
    diff_sub.add_argument("--use_columns", nargs=2, default=[0,0], type=int, help="Specify which columns of the data to load")
    
    #group1 = diff_sub.add_mutually_exclusive_group("skiprows", "Specify the number of rows to skip when loading a file. Several ways of doing this is shown below:")
    #group1.add_argument("--skiprows", type=int, nargs=2, help="Specify the number of rows to skip in each file individually, takes two arguments")
    #group1.add_argument("--skipboth", type=int, help="Specify the number of rows to skip in each file individually")
    
    
    #same_sub for analyzing a single file containing the DCs in two separate columns
    same_sub = sub.add_parser("same", parents=[par], help="For using data from a single file")
    same_sub.add_argument("--dc_file", type=str, help="Specify the file containing the DC coordinates")
    same_sub.add_argument("--dc_use", type=int, nargs=2, default=[1, 2], help="specify which DC coordinates to use, defaults are 1 and 2")
    same_sub.add_argument("--skiprows", type=int, default=0, help="Specify the number of rows to skip when reading the file")
    
    
    args = parser.parse_args()
    args = sanitize_args(args)

    
    return args

def sanitize_args(args):
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    if args.sub_type == "same": 
        if not os.path.isfile(args.dc_file):
            raise IOError("Cannot find DC files, aborting")
    
    return args
    
if __name__=="__main__":
    cwd = os.getcwd()
    args = get_args()
    
    dcA, dcB = get_DCs(args)
    
    x,y,z, slices = histogram_DCs(dcA, dcB, args.bins, args.range, args.temp, args.smooth)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    qmesh = ax.pcolormesh(x,y,z.transpose()) #Note, the transpose is taken because pyplot follows the matlab convention. Columns of the matrix correspond to the x-axis and rows correspond to the y-axis. This is different from the np.histogram2d function, as the rows correspond to the x-edges and columns correspond to the y-edges
    plt.xlabel("DC1")
    plt.ylabel("DC1")
    plt.colorbar(qmesh)
    
    os.chdir(args.save_dir)
    plt.savefig("%s.png" % args.save_name)
    
    if args.interactive:
        mode = ModeSelect(ax, dcA, dcB, slices, x, y, append=args.append)
        plt.show(fig)
    
    os.chdir(cwd)
    
    
    
    
    
    
    
    
