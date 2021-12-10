#!/usr/bin/python

"""
Module for fitting 3D spine shapes to 2D outlines

Revision Date: 2017-02-20
Copyright University of Michgan
Contact: Matt Reed (mreed@umich.edu)

Python Version: 3.5


Usage:

# python CSpine3DFitting.py

The output directory is expected to contain a file called SpineOut.tsv containing the 2D landmarks. 
An alternative file can be supplied on the command line:

# python CSpine3DFitting.py AlternativeSpine.tsv

Note that the path for the alternative file is relative to the module.
The data directory containing the bone mesh and landmark files must be in the same directory as the module.

"""


__author__ = "Matt Reed"
__copyright__ = "Copyright 2017 Regents of the University of Michigan"
__version__ = "2017-002-2"
__email__ = "mreed@umich.edu"


import numpy as np
from scipy import spatial
import os, math, copy, sys, csv


BONE_NAMES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
MEAN_FILE_NAME = "_mean.obj"

DATA_DIRECTORY = "data/"
OUTPUT_DIRECTORY = "output/"

# input file is output from 2D model
INPUT_FILE_NAME = OUTPUT_DIRECTORY+"SpineOut.tsv"
LANDMARK_VERTEX_POSITION_FILE_NAME = 'landmark_vertex_positions.tsv'
LANDMARK_VERTEX_NAMES_FILE_NAME = 'landmark_vertex_names.tsv'

class OBJ(object):
    """ Load, store, and save OBJ file data """
    
    def __init__(self, filename=None, file_type='auto'):
        self.filename = filename
        self.verts = None # will be np.array
        self.polys = []
        if self.filename:
            self.load_obj(self.filename)
        else:
            self.filename = 'out.obj'
                
                
    def scale(self, value, center=True):
        """ Scale the verts by value."""
        if center:
            mean_pt = np.mean(self.verts, axis=0)
            self.verts = (self.verts-mean_pt)*value + mean_pt
        else:
            self.verts *= value
            
    def translate(self, trans):
        """ Translate the verts by trans, which should be length 3."""
        self.verts += np.asarray(value)
        
        
    def rotate(self, m, center=True):
        """ Rotate around global origin or mean by 3x3 matrix m."""
        
        if center:
            org = np.mean(self.verts, axis=0)
        else:
            org = np.array([0,0,0])
        
        self.verts = (self.verts - org).dot(m) + org
    
    def copy(self):
        """ Return a copy. """
        new_obj = OBJ()
        new_obj.filename = self.filename
        if self.verts is not None:
            new_obj.verts = self.verts.copy()
        if self.polys:
            new_obj.polys = copy.deepcopy(self.polys)
            
        return new_obj
        
    
    def load_obj(self, name, flip_polys=True):
        """ Load the file with the specified name, return [verts, polys].
        Color and texture are ignored, as are groups. """
        verts = []
        polys = []
        
        with open(name, 'r') as f:
            for line in f.readlines():
                if line.startswith('v') and not line.startswith('vn'):
                    line_list = line.split()[1:]
                    list_list_coord = [v.split("/")[0] for v in line_list]
                    verts.append(np.array([float(a) for a in list_list_coord]) )
                elif line.startswith('f'):
                    line_list = line.split()[1:]
                    list_list_coord = [v.split("/")[0] for v in line_list]
                    polys.append([int(a) for a in list_list_coord]) 
        
        self.verts = np.array(verts)
        self.polys = polys
        
        if flip_polys:
            for p in self.polys:
                p.reverse()

    def save_obj(self, filename=None, filename_extra='_out'):
        """ Save as OBJ to file name. """
        out_filename = filename
        if not out_filename:
            out_filename = self.filename[:-4]+filename_extra+".obj"
        
        print("Saving ", out_filename)
        
        with open(out_filename, 'w') as f:
            f.write("# comment\n")
            for v in self.verts:
                f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
            for vlist in self.polys:
                f.write('f')
                for v in vlist:                    
                    f.write(' {}'.format(v))
                f.write('\n')

        
class Spine2D(object):
    """ Load & store a 2D spine from a file """
    def __init__(self, filename):
        self.filename = filename
        self.landmarks = {} # store landmarks as dict
        self.load_landmarks(self.filename )
        
        
    def load_landmarks(self, filename):
        """ Load a landmark file. """
        
        self.landmarks = {}
        with open(filename, 'r') as f:
            for line in f.readlines():
                line_in = line.split()
                if len(line_in) == 3:
                    self.landmarks[line_in[2]] = np.array([float(v) for v in line_in[0:2]])
            

        return self.landmarks   
        
        
class Bone3D(object):
    """ store info about a bone """
    
    def __init__(self, filename=None, name=None):
        
        self.filename = filename
        self.name = name
        self.geometry = OBJ(filename)
        self.landmark_names = []
        self.landmarks = {}
        self.landmark_vertices = []
        
    def assign_landmark(self, name, vertex_index):
        """ assigns a landmark based on name and index into the vertex list """
        
        self.landmarks[name] = self.geometry.verts[vertex_index]
        self.landmark_names.append(name)
        self.landmark_vertices.append(vertex_index)

    def assign_landmarks(self, names, vertex_indices):
        """ assigns a landmark based on name and index into the vertex list """
        
        for n, v in zip(names, vertex_indices):
            self.assign_landmark(n, v)
    
    def export(self):
        """ export geometry """
        self.geometry.save_obj(OUTPUT_DIRECTORY+self.name+"_out.obj")
        
    def export_landmarks(self, filename="_land_out.tsv"):
        """ Write landmarks to the specified file. """
        land = self.landmarks
        with open(OUTPUT_DIRECTORY+self.name+filename, 'w', newline='') as file_out:
            writer = csv.writer(file_out, delimiter="\t")
            for n in self.landmark_names:
                try:
                    line = [float('%.3f'%v) for v in land[n]]
                    line.append(n)
                    writer.writerow(line)
                except KeyError:
                    # print("Can't write ", n)
                    pass
        
    
class BoneModel3D(object):
    """ Load and store PCA results for spine bones """
    
    
    def __init__(self):
        
        
        # get an array of bones    
        self.bones = [Bone3D(DATA_DIRECTORY+bn+MEAN_FILE_NAME, bn) for bn in BONE_NAMES]
        
        # get landmark vertices and assign
        self.landmark_vertices = self.load_landmark_vertices()
        self.landmark_names = self.load_landmark_names()
        [b.assign_landmarks(n, v) for b, n, v in zip(self.bones, self.landmark_names, self.landmark_vertices)]
                
        
    def load_landmark_vertices(self, fn=LANDMARK_VERTEX_POSITION_FILE_NAME):
        """ load the vertex information associated with the bones """
        
        vertex_indices = []
        with open(DATA_DIRECTORY+fn, 'r') as f:
            for line in f.readlines():
                vertex_indices.append([int(v)-1 for v in line.split()]) # NOTE! shifting to zero as first index!
                
        return vertex_indices    

    def load_landmark_names(self, fn=LANDMARK_VERTEX_NAMES_FILE_NAME):
        """ load the landmark names associated with the bones """
        landmark_names = []
        with open(DATA_DIRECTORY+fn, 'r') as f:
            for line in f.readlines():
                landmark_names.append(line.split())
                
        return landmark_names    
        
    def export(self):
        """ export all the bones """
        for b in self.bones:
            b.export()
            b.export_landmarks()
            

class BoneMapper(object):
    """ class to map 3D bones to 2D outlines """
    
    def __init__(self, filename=None, export=True):
        
        # get 2D spine
        fn = filename if filename else INPUT_FILE_NAME
        
        self.spine2d = Spine2D(fn)
        self.spine3d = BoneModel3D()
        
         #  mapping points are the first n landmarks per bone
        self.num_map_pts = [2, 6, 8, 8, 8, 8, 8]
         # for each bone, get the names of the landmarks to use
        self.landmarks_to_use = [b.landmark_names[0:n] for b, n in zip(self.spine3d.bones, self.num_map_pts)]
        
        # map the bones
        for b, lm in zip(self.spine3d.bones, self.landmarks_to_use):
            self.map_bone(b, self.spine2d, lm)
                
        if export:
            self.spine3d.export()
        
    def map_bone(self, bone3d, spine2d, landmark_names):
        """ map a 3d bone to the corresponding landmarks in spine2d """
        
        # algorithm: 
        # 1. center landmarks
        # 2. get centroid size and unscale landmarks
        # 3. get optimal aligment rotation matrix
        # 4. scale, rotate, and align bone 3D vertices
        
        # store intermediate results in the bone3d
        
        # make landmark arrays
        bone3d.target_landmarks = np.array([spine2d.landmarks[land] for land in landmark_names])
        bone3d.current_landmarks = np.array([bone3d.landmarks[land][[0,2]] for land in landmark_names]) # make 2D
        
        # compute mean values
        bone3d.target_mean = np.mean(bone3d.target_landmarks, 0)
        bone3d.current_mean = np.mean(bone3d.current_landmarks, 0)
        
        # center landmarks
        bone3d.target_landmarks_centered = bone3d.target_landmarks - bone3d.target_mean
        bone3d.current_landmarks_centered = bone3d.current_landmarks - bone3d.current_mean
        
        # compute target size and scale factor
        bone3d.target_size = np.sqrt(np.trace(bone3d.target_landmarks_centered.dot(np.transpose(bone3d.target_landmarks_centered))))
        bone3d.current_size = np.sqrt(np.trace(bone3d.current_landmarks_centered.dot(
                np.transpose(bone3d.current_landmarks_centered))))
        
        sf = bone3d.target_size/bone3d.current_size
        # print("scale factor: ", sf)
        
        # unscale centered data
        bone3d.target_landmarks_centered_scaled = bone3d.target_landmarks_centered/bone3d.target_size
        bone3d.current_landmarks_centered_scaled = bone3d.current_landmarks_centered/bone3d.current_size
                
        # compute least-squares rotation matrix using singular value decomposition
        u, s, v = np.linalg.svd(bone3d.target_landmarks_centered_scaled.transpose().dot(
            bone3d.current_landmarks_centered_scaled))        
        bone3d.rm = v.transpose().dot(u.transpose())       # rotation matrix  

        # result may include a reflection as well as a rotation; correct if so
        if np.linalg.det(bone3d.rm) < 0:
            bone3d.rm *= np.array([1,-1])
            
        # now apply scale, rotation, and translation to vertices of bone geometry
        bone3d.geometry.verts = self.apply_transformation3d(bone3d.geometry.verts, bone3d.current_mean, bone3d.target_mean, bone3d.rm, sf)

        # transform all landmark, not just the ones used for alignment
        for lm in bone3d.landmarks.keys():
             bone3d.landmarks[lm] = self.apply_transformation_land(
                np.array(bone3d.landmarks[lm]), bone3d.current_mean, bone3d.target_mean, bone3d.rm, sf)
           
        
        bone3d.new_landmarks = np.array([bone3d.landmarks[land][[0,2]] for land in landmark_names]) # make 2D
        
        bone3d.rmse = self.compute_rmse(bone3d.new_landmarks, bone3d.target_landmarks)
        print(bone3d.name, "rmse: ", bone3d.rmse)
    
    def apply_transformation3d(self, vert3d, center, trans, rotation, scale):
        """ applies the planar transformation to the 3d vert array """
            
        v2d = vert3d[:,(0,2)] # get x z only
        v2dout = ((v2d-center)*scale).dot(rotation) + trans

        return np.insert(v2dout, 1, vert3d[:,1], axis=1) # reinsert Y axis

    def apply_transformation_land(self, vert3d, center, trans, rotation, scale):
        """ applies the planar transformation to the 3d vert array """
        
        v2d = vert3d[[0,2]]
        v2dout = ((v2d-center)*scale).dot(rotation) + trans
        return np.insert(v2dout, 1, vert3d[1])


    def apply_transformation2d(self, vert2d, center, trans, rotation, scale):
        """ applies the planar transformation to the 2d vert array """
        
        return ((vert2d-center)*scale).dot(rotation) + trans

    def compute_rmse(self, v1, v2):
        """ computes the root-mean-square distance error between the corresponding vertices """
        
        return np.sqrt(np.mean(np.linalg.norm(v1-v2, axis=1)**2))

# convenience functions

def normalize(v):
    return v/np.linalg.norm(v)
    
def distance(p1, p2):
    return np.linalg.norm(p2-p1)
    
# run
  
if __name__ == '__main__': # when run as module

    input_file = None
    if len(sys.argv) > 1:  # interpret argument as input file name for 2D spine
        input_file = sys.argv[1]
    
    bm = BoneMapper(input_file)
     
            

    