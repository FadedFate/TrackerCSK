#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is a python reimplementation of the open source tracker in
http://www2.isr.uc.pt/~henriques/circulant/index.html

Found http://wiki.scipy.org/NumPy_for_Matlab_Users very useful

Based on the work of João F. Henriques, 2012
http://www.isr.uc.pt/~henriques

Rodrigo Benenson, MPI-Inf 2013
http://rodrigob.github.io
"""

from __future__ import print_function

import os
import os.path
import sys
import glob
import time
from optparse import OptionParser
import time
from got10k.trackers import Tracker
import numpy as np

import ops
import scipy.misc
import pylab

debug = False

__all__ = ['CSK_tracker']

class TrackerCSK(Tracker):

    def __init__(self, padding = 0.5 , sigma = 0.2 , output_sigma_factor = 1/float(16) , ):
        """
        object_example is an image showing the object to track
        """
        super(TrackerCSK, self).__init__('CSK', True)
        self.padding = padding
        self.sigma = sigma
        self.output_sigma_factor = output_sigma_factor
        self.lambda_value = 1e-2  # regularization
        self.interpolation_factor = 0.075

    def init(self, img, box):
        img_now = ops.read_image(img)
        self.target_sz = np.array([box[3] , box[2]])
        self.pos = np.array([box[1] , box[0]]) + self.target_sz / 2
        # print(self.pos)
        # ground_truth = 

        # window size, taking padding into account
        self.sz = pylab.floor(self.target_sz * (1 + self.padding))

        # desired output (gaussian shaped), bandwidth proportional to target size
        self.output_sigma = pylab.sqrt(pylab.prod(self.target_sz)) * self.output_sigma_factor

        grid_y = pylab.arange(self.sz[0]) - pylab.floor(self.sz[0]/2)
        grid_x = pylab.arange(self.sz[1]) - pylab.floor(self.sz[1]/2)
    #[rs, cs] = ndgrid(grid_x, grid_y)
        rs, cs = pylab.meshgrid(grid_x, grid_y)
        y = pylab.exp(-0.5 / self.output_sigma**2 * (rs**2 + cs**2))
        self.yf = pylab.fft2(y)
        # print(self.yf)
    #print("yf.shape ==", yf.shape)
    #print("y.shape ==", y.shape)

    # store pre-computed cosine window
        self.cos_window = pylab.outer(pylab.hanning(self.sz[0]),
                             pylab.hanning(self.sz[1]))
        if img_now.ndim == 3:
            img_now = ops.rgb2gray(img_now)
        x = ops.get_subwindow(img_now, self.pos, self.sz, self.cos_window)
        k = ops.dense_gauss_kernel(self.sigma, x)
        self.alphaf = pylab.divide(self.yf, (pylab.fft2(k) + self.lambda_value))  # Eq. 7
        self.z = x
        # print(self.z)
        # print(self.alphaf)
    def update(self, img):
        img_now = ops.read_image(img)
        if img_now.ndim == 3:
            img_now = ops.rgb2gray(img_now)
        x = ops.get_subwindow(img_now, self.pos, self.sz, self.cos_window)
        # print(x)
        k = ops.dense_gauss_kernel(self.sigma, x, self.z)
        kf = pylab.fft2(k)
        alphaf_kf = pylab.multiply(self.alphaf, kf)
        response = pylab.real(pylab.ifft2(alphaf_kf))  # Eq. 9
        
        # target location is at the maximum response
        r = response
        row, col = pylab.unravel_index(r.argmax(), r.shape)
        
        self.pos = self.pos - pylab.floor(self.sz/2) + [row, col]
        x = ops.get_subwindow(img_now , self.pos , self.sz , self.cos_window)
        k = ops.dense_gauss_kernel(self.sigma, x)

        new_alphaf = pylab.divide(self.yf, (pylab.fft2(k) + self.lambda_value))  # Eq. 7
        new_z = x
        f = self.interpolation_factor
        self.alphaf = (1 - f) * self.alphaf + f * new_alphaf
        self.z = (1 - f) * self.z + f * new_z
        
        box_new = np.array([self.pos[1]-(self.sz[1])/2 + 1 , self.pos[0]-(self.sz[0])/2 + 1 , self.sz[1] , self.sz[0] ] ,dtype = np.float32)
        return box_new

    def track(self, img_files, box , visualize = False) :
        frame_num = len(img_files)
        
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)
        for f, img_file in enumerate(img_files):
            begin = time.time()
            if f == 0:
                self.init(img_files[0] , box)
            else:
                boxes[f, :] = self.update(img_file) 
            times[f] = time.time() - begin  
        return boxes, times
