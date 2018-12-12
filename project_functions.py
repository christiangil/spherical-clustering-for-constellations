# project_functions.py

# importing packages

# data manipulation
import numpy as np
import pandas as pd

# benchmarking
import time

# importing and saving analysis data
import os.path

# letting me know when large calculattions are done
import winsound

# plotting and image exporting
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

# importing star data from VizieR
data = pd.read_csv('data_big.csv')
data = data[data["Hpmag"]<6]
n_stars = len(data)


# creating a basic normalization function which sets the minimum entry to zero and the maximum entry to one
def normalize(a):
    if np.min(a) != np.max(a):
        return (a - np.min(a)) / (np.max(a) - np.min(a))
    else:
        return a


# getting star positions from the data
star_ra = data["Radeg"] * np.pi / 180
star_dec = data["Dedeg"] * np.pi / 180
# converting right ascention and declination to cartesian coordinates on unit celestial sphere
# conversion from https://en.wikipedia.org/wiki/Equatorial_coordinate_system
star_x = np.cos(star_dec) * np.cos(star_ra)
star_y = np.cos(star_dec) * np.sin(star_ra)
star_z = np.sin(star_dec)
pos = pd.concat([star_x, star_y, star_z], axis=1)
pos = pos.rename(columns = {0:"x", 1:"y", "Dedeg":"z"})

# star brightnesses from data
star_mag = data["Hpmag"]
dark = normalize(star_mag)
# a normalized array setting the largest brightnesses to be 0.9 and the smallest to be 0 ends up creating a factor of 10 between the darkest and brightest stars
bright = np.array(0.9 * np.array(1 - dark))


# numpy version of converting right ascention and declination to cartesian coordinates on unit celestial sphere
# conversion from https://en.wikipedia.org/wiki/Equatorial_coordinate_system
def ra_dec_2_cart(ra, dec):
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


# importing centers and areas of the modern constellations
# data from https://en.wikipedia.org/wiki/88_modern_constellations_by_area
truth = pd.read_csv("constellation_centers.csv")
true_centers = np.zeros((88, 3))
for i in range(88):
    ra0 = truth["RA"].iloc[i]
    dec0 = truth["Dec"].iloc[i]
    true_centers[i, :] = ra_dec_2_cart(ra0, dec0)
percent_const = truth["SA"] / 41253  # percent of sky that each constellation takes up
percent_const /= np.sum(percent_const)  # make sure it normalizes to one


# calculates the sum of (brightness-weighted) distances of a point on the celestial sphere to all of the stars
def weighted_cosine_dists(center, weights=np.zeros(n_stars)):
    dists = np.zeros(n_stars)
    for i in range(n_stars):
        val = np.dot(center, pos.iloc[i])
        # prevents errors with np.arccos()
        if val > 0.9999:
            dists[i] = 0.
        else:
            # dists[i] = dark.iloc[center] * dark.iloc[i] * np.arccos(val)
            # dists[i] = np.arccos(val)  # spherical separation
            dists[i] = (1 - weights[i]) * (1 - val) # cosine dissimilarity, darker things should be seen as further away
    return dists


# numpy array version of converting right ascention and declination to cartesian coordinates on unit celestial sphere
# conversion from https://en.wikipedia.org/wiki/Equatorial_coordinate_system
def ra_dec_2_cart_array(ra, dec):
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.stack((x, y, z), axis=1)


# formula for a normalized spherical gaussian
# from https://mynameismjp.wordpress.com/2016/10/09/sg-series-part-2-spherical-gaussians-101/
def sgauss(x, mu=np.array([1, 0, 0]), lam=1, a=1):
    return (a * lam / (2 * np.pi * (1 - np.exp(-2 * lam)))) * np.exp(lam * (np.dot(mu, x) - 1))


# evaluates multiple spherical gaussians at x with weights "a", centers "mus", and inverse widths "lams"
def sgauss_tot(x, mus, lams=1., a=1.):
    tot = 0
    for i in range(np.shape(mus)[0]):
        if type(a) is float:
            a_cur = a
        else:
            a_cur = a[i]
        if type(lams) is float:
            lam_cur = lams
        else:
            lam_cur = lams[i]
        tot += sgauss(x, mu=mus[i, :], lam=lam_cur, a=a_cur)
    return tot


# create a 3D figure (for cool plots)
def init_3D_figure(figsize=[16, 16]):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax


# calculate the cluster centers and cluster identities of stars with k-center method
# k-center only needs to evaluate pairwise distances once per amount of clusters which makes it fast
def k_center(n_const=88, n_stars=n_stars, weights=np.ones(n_stars)):

    # initialize cluster centers and cluster identities matrices
    cluster_total = np.zeros((1, n_stars))
    centers = np.zeros((n_const, 3))
    
    # start time (for benchmarking)
    start = time.time()
    
    # randomly initialize the first cluster at the position of one of the stars
    centers[0, :] = pos.iloc[np.random.randint(0, high=n_stars)]

    # calculate the distance to the initial cluster
    dists = weighted_cosine_dists(centers[0, :], weights=weights)
    
    # the amount of desired clusters - 1
    for i in range(1, n_const):

        # create the next row of the cluster identity matrix with a shape that is ammenable to appending later
        cluster_temp = np.ones((1, n_stars))
        cluster_temp[0, :] = cluster_total[-1, :]

        # choose the star that is furthest away from the other clusters to be the center of a new cluster
        centers[i, :] = pos.iloc[np.argmax(dists)]

        # calculate distances from all stars to the new center
        dists_temp = weighted_cosine_dists(centers[i, :], weights=weights)
        
        # for every star
        for j in range(n_stars):
            
            # if the new cluster center is closer than its previous assignment
            if dists_temp[j] < dists[j]:

                # set the new cluster identity and distance to cluster center
                cluster_temp[0, j] = i
                dists[j] = dists_temp[j]

        # append the new cluster identities to the end of the cluster identity matrix to keep a history of cluster identities
        cluster_total = np.append(cluster_total, cluster_temp, axis=0)
    
    # end time (for benchmarking)
    end = time.time()
    print("clustering took %f seconds"%(end - start))

    # return cluster history and centers
    return cluster_total, centers


# get random vectors on the unit sphere
def random_unit_sphere(n=1):

    # initialize draws
    draws = np.zeros((n, 3))
        
    # for the amount requested
    for i in range(n):
        
        # only accept a random vector if it is less than 1 in magnitude (ensures even spherical sampling)
        length = 2
        while length > 1:
            draw = np.random.uniform(low=-1.0, high=1.0, size=3)
            length = np.dot(draw, draw)
        
        # record the random unit vector
        draws[i, :] = draw / np.linalg.norm(draw)

    # record the random unit vectors
    return draws


# calculate the cluster centers and cluster identities of stars with spherical k-means method
def k_means(n_const=88, n_stars=n_stars, weights=np.ones(n_stars), online=False, tol = 1e-3):
    
    # initialize cluster centers and cluster identities matrices
    cluster_total = np.zeros((1, n_stars))
    centers_total = np.zeros((1, n_const, 3))
    
    # start time (for benchmarking)
    start = time.time()

    # inialize the cluster centers randomly
    centers_total[0, :, :] = random_unit_sphere(n=n_const)

    # used to see the relative change in total distance from cluster centers
    # a measure of convergence
    delta = 1

    # initial distances of stars to cluster 0 
    dists = weighted_cosine_dists(centers_total[0, 0, :], weights=weights)

    # a default learning rate
    # only used with online (non-batch) learning methods
    eta0 = 1/(5000/88)

    # used to modify the learning rate in online learning "anneal"ing scheme
    counter = 0
    batch_iterations = 15

    # until relative change in distance is less than the tolerance
    while delta > tol:

        ## assign to updated cluster centers

        # get total distances of stars to their nearest clusters
        dist_tot_1 = np.sum(dists)

        # get last cluster assignments
        cluster_temp = np.zeros((1, n_stars))
        cluster_temp[0, :] = cluster_total[-1, :]

        # get last cluster centers
        centers_temp = np.zeros((1, n_const, 3))
        centers_temp[0, :, :] = centers_total[-1, :, :]

        # for each cluster
        for i in range(n_const):

            # find distance of all stars to the cluster center
            dists_temp = weighted_cosine_dists(centers_temp[0, i, :], weights=weights)

            # for each star
            for j in range(0, n_stars):

                # if the star is closer to this cluster than its pervious one, move it to the new cluster and record its new distance
                if dists_temp[j] < dists[j]:
                    cluster_temp[0, j] = i
                    dists[j] = dists_temp[j]

        # qppend the new cluster identities to the history matrix
        cluster_total = np.append(cluster_total, cluster_temp, axis=0)
        
        # find new some of distances
        dist_tot_2 = np.sum(dists)

        # see if it has changed enough that we have converged
        delta = abs(dist_tot_1 - dist_tot_2) / dist_tot_1
        print(delta)
        
        ## calculate new cluster centers after stars have been reassigned

        # if using online learning
        if type(online) is str:

            # for each star
            for i in range(n_stars):

                # for its cluster
                current_assign = int(cluster_temp[0, i])

                # set learning rate based on method requested
                if online=='anneal':
                    eta = 5 * eta0 * (1 / 10) ** (counter / (n_stars * batch_iterations))  # an exponential learning rate
                    counter +=1
                elif online=='balance':
                    eta = 1 / np.sum(np.where(cluster_temp[0, :] == current_assign))  # a balancing learning rate
                elif online == 'flat':
                    eta = eta0  # a flat learning rate
                else:
                    print("improper online learning type assigned")

                # move the cluster center towards the cluster member and renormalize
                hold = centers_temp[0, current_assign, :] + eta * (0.1 + weights[i]) * pos.iloc[i]
                centers_temp[0, current_assign, :] = hold / np.linalg.norm(hold)

            # add the new center to the history matrix
            centers_total = np.append(centers_total, centers_temp, axis=0)
        
        # if using batch learning  
        else:
            
            # set the cluster center to be the place that minimizes the L1 norm of the distances of the cluster members to the cluster center
            # only works for cosine dissimilarity
            for i in range(n_const):
                ind = np.where(cluster_total[-1, :] == i)
                # this could be reworked
                sx = np.array([np.dot(pos["x"].iloc[ind], (0.1 + weights[i])), np.dot(pos["y"].iloc[ind], (0.1 + weights[i])), np.dot(pos["z"].iloc[ind], (0.1 + weights[i]))])
                centers_temp[0, i, :] = sx / np.linalg.norm(sx)

            # add the new center to the history matrix
            centers_total = np.append(centers_total, centers_temp, axis=0)
        
        # end time (for benchmarking)
        end = time.time()
        print("clustering has taken %f seconds"%(end - start))
    
    
    ## final cluster assignment
    cluster_temp = np.ones((1, n_stars))
    cluster_temp[0, :] = cluster_total[-1, :]
    centers_temp = np.zeros((1, n_const, 3))
    centers_temp[0, :, :] = centers_total[-1, :, :]
    for i in range(n_const):
        dists_temp = weighted_cosine_dists(centers_temp[0, i, :], weights=weights)
        for j in range(0, n_stars):
            if dists_temp[j] < dists[j]:
                cluster_temp[0, j] = i
                dists[j] = dists_temp[j]
    cluster_total = np.append(cluster_total, cluster_temp, axis=0)
    
    # end time (for benchmarking)
    end = time.time()
    print("clustering took %f seconds"%(end - start))
    
    # final cluster centers
    centers_final = centers_total[-1, :, :]

    return cluster_total, centers_total, centers_final


# get a numpy array from a numpy file if it exists
def array_from_file(filename):

    if os.path.isfile(filename):
        array = np.load(filename)
    else:
        array = np.array([])
    return array


# repeatedly calculate new cluster centers, see how close they are to the normal constellations, and record how close they are in a numpy file
def bootstrap_array(resample=0, filename=None, n_const=88, cluster_func="random", use_weights=False, online=False, updates=False, alert=True, lams=100., save_iteration=True):
    
    # get the previous results
    boot_array = array_from_file(filename)
    
    # if you are requesting more scores
    if resample != 0:

        # for each new score
        for i in range(resample):

            # tell them which iteration we are on
            if updates:
                print("working on bootstrap %d"%(1+i) + " for " + cluster_func + " clustering method")
            
            # initialize the score at 0
            hold = 0.

            # get new cluster centers
            centers_final = evaluation_clustering(cluster_func, use_weights=use_weights, n_const=n_const, online=online)
            
            # add the score of each center based on how close it is to normal constellation centers
            for j in range(n_const):
                hold += sgauss_tot(centers_final[j, :], true_centers, lams=lams, a=percent_const)\

            # append the new score to the old array
            boot_array = np.append(boot_array, hold)
            
            # save the old + new results after each new score is calculated in case something goes wrong 
            # dont do this if your clustering method is too fast, you will try to write to the file while it is still open from a previous run
            if save_iteration:
                np.save(filename, boot_array)
        
        # save the old + new results to the same file
        np.save(filename, boot_array)
        
        # beep to let me know this batch is done
        if alert:
            # setting sounds of alert beep
            frequency = 262  # Set Frequency (Hz), middle C
            duration = 1000  # Set Duration To 1000 ms == 1 second
            winsound.Beep(frequency, duration)
        
    return boot_array


# clustering function wrapper only returning the final centers
def evaluation_clustering(cluster_func, use_weights=False, n_const=88, online=False):
    
    # set weights based on use_weights argument
    if use_weights:

        # weight by brightness
        weights = bright
    
    else:
        weights = np.ones(n_stars)
    

    # use kmeans if specified
    if cluster_func == "kmeans":
        cluster_total, centers_total, centers_final = k_means(n_const=n_const, weights=weights, online=online)
    
    # use kmeans if specified
    elif cluster_func == "kcenter":
        cluster_total, centers_final = k_center(n_const=n_const, weights=weights)
    
    # create random vectors as the centers if any other method is requested
    else:
        centers_final = random_unit_sphere(n=n_const)

    # return the proper, final cluster centers
    return centers_final


# initialize a large 2D figure with large labels. wide affects aspect ratio. plot_size changes the overall size
def init_plot(plot_size=1, wide=1):
    
    # create figure
    fig = plt.figure(figsize=[16 * plot_size * wide, 16 * plot_size])
    
    # make labels easy to work with
    ax = plt.subplot(111)

    # change label sizes
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(40 * plot_size)

    # return ax so the labels can be set
    return ax


# show the plot and save it if they want
def finish_plot(fig_name=""):

    # show plot
    fig = plt.gcf()

    # save it at the specified location
    if fig_name!="":
        fig.savefig(fig_name, bbox_inches="tight")