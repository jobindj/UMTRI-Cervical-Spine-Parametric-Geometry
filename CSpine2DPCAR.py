#!/usr/bin/python

"""
Module for generating and flexing/extending 2D C-spine outlines.

Revision Date: 2017-02-25
Copyright University of Michgan
Contact: Matt Reed (mreed@umich.edu)

Python Version: 3.5

Typical usage:
    
>>> from CSpine2DPCAR import *
    
>>> target_anthro = anthro(anthro.FEMALE, stature=1650, age=45, shs=0.52) # shs = sitting height / stature 
>>> pcar = PCARSpine2D()
>>> pcar.predict(target_anthro, delta_head_angle=-30) # delta head angle from neutral in degrees; negative is flexion
    
The predict() methods generates a spine using the PCAR model,
articulates the model according to the delta_head_angle, 
then writes the model to a file "SpineOut.tsv" that contains named points.

Opening SpineOut.tsv in Excel will automatically update the plot in ViewSpine.xlsx.

The PCARSpine2D.predict() method can also be called with keyword arguments:

>>> pcar.predict(sex=anthro.FEMALE, stature=1750, age=45, delta_head_angle=20)

or with a list of anthro

>>> pcar.predict(anthro=[1, 1750, 0.52, 45], delta_head_angle=20) # sex: -1=male, 1=male

The module can also be run from the command prompt with the anthro and posture as command-line arguments

$ python CSpine2DPCAR.py 1 1650 0.52 20 0

Note all five arguments (sex, stature, shs, age, and delta head angle) must be supplied. If none is supplied, a midsize female
spine is generated in the neutral posture.

A location parameter can be added to the predict() call to translate and rotate the model. 
location='C7' (or other level up to C2) will place the anterior-inferior margin of the body at the origin and align the inferior surface
of the body with the global x axis. Alternatively, enter a location and angle, e.g., [[40, 40], 30] will
translate the model by [40, 40] and rotate 30 degrees clockwise.

The segment positions and orientations are written at the end of the landmark file.

"""

__author__ = "Matt Reed"
__copyright__ = "Copyright 2017 Regents of the University of Michigan"
__version__ = "2017-02-25"
__email__ = "mreed@umich.edu"


import csv, sys
import numpy as np
from scipy import interpolate
from scipy.special import binom  # used in Bernstein polynomials for spline fitting
from scipy.optimize import minimize_scalar # used to find points at distances along spline

DATA_DIRECTORY = "data/"
OUTPUT_DIRECTORY = "output/"

PC_MATRIX_FILENAME = DATA_DIRECTORY+"PCMatrix.tsv"
# REGRESSION_COEFFICIENTS_FILENAME = DATA_DIRECTORY+"RegressionCoefficients.tsv"
REGRESSION_COEFFICIENTS_FILENAME = DATA_DIRECTORY+"RegressionCoefficients_sex+interactions.tsv"
MEAN_MODEL_FILENAME = DATA_DIRECTORY+"MeanModel.tsv"
EXPORT_FILENAME = OUTPUT_DIRECTORY+"SpineOut.tsv"


class SpineModel2D(object):
    """ class defining a 2D c-spine model defined by landmarks """
    
    
    # these are the point names that are used to control the posture of each segment
    # the segment orientation is given by the first two and the joint to rotate around by the third
    # note that "C7Joint" refers to the C7/T1 joint, etc.
   
    #spline_fit_points =  ["C2_SupMidDen", "C2Joint", "C3Joint", "C4Joint", "C5Joint", 
    #    "C6Joint", "C7Joint"]
        
    spline_fit_points = ["C2_SupMidDen", "C2RotationCenter", "C3RotationCenter", \
            "C4RotationCenter", "C5RotationCenter", "C6RotationCenter", \
            "C7RotationCenter"]

    # points used to find "joints" at intervertebral space
    joint_points = [["C2_InfMedBod", "C3_SupMedBod", "C2Joint"],
        ["C3_InfMedBod", "C4_SupMedBod", "C3Joint"],
        ["C4_InfMedBod", "C5_SupMedBod", "C4Joint"],
        ["C5_InfMedBod", "C6_SupMedBod", "C5Joint"],
        ["C6_InfMedBod", "C7_SupMedBod", "C6Joint"]]
        
    # these are the relationships used to find the joint centers 
        
    # new version based on Amevo 1991; uses local coords from ant-inf
    # four points defining the boundary of the lower body, then two fractions
    
    icrFractions = [["C2RotationCenter", "C3", 0.27, 0.36],
    
    ["C3RotationCenter", "C4",  0.32, 0.52],
    ["C4RotationCenter", "C5", 0.36, 0.60],
    ["C5RotationCenter", "C6", 0.39, 0.78],
    ["C6RotationCenter", "C7", 0.44, 0.95]
    ] # handle C7 separately -- use C7Joint 

    # alternative motion distribution based on Snyder. Take the mean head vs. T1 ROM of 117 deg, 
    # divide among the motion segments by fraction of total ROM:
    # head, C2C3.... C7T1
    
    romFractions = [0.192, 0.038, 0.104, 0.197, 0.181, 0.160, 0.128]
    
        
    vertebra_positioning_list = [[["C2_SupMidDen", "C2RotationCenter"],["C2_SupAntDen", "C1_AntSupArc", 
        "C1_AntTub", "C1_AntInfArc", "C1C2_AntInt", "C1C2_PosInt", 
        "C1C2_PosInt", "C1_PosInfArc", "C1_InfMidArc", "C1_InfCan", 
        "C1_SpiPro", "C1_SupCan", "C1_SupMidArc", "C1_PosSupArc", 
        "C2_SupPosDen", "C2_SupAntDen", "C2_MedAntFce", "C2_AntInfBod", 
        "C2_InfMedBod", "C2_PosInfBod", "C2_AntInfFac", "C2_PosInfFac", 
        "C2_InfCan", "C2_InfSpiPro", "C2_InfSpiPro", "C2_SupSpiPro", 
        "C2_SupCan", "C2_PosSupFac", "C2_AntSupFac", "C2_SupPosDen", 
        "C2_SupPosDen", "C2_SupMidDen", "C2_SupAntDen", "HeadRotationCenter"]],
        [["C2RotationCenter", "C3RotationCenter"],
        ["C3_AntInfBod", "C3_AntMedBod", "C3_AntSupBod", 
        "C3_SupMedBod", "C3_PosSupBod", "C3_AntSupFac", "C3_PosSupFac", 
        "C3_SupCan", "C3_SupSpiPro", "C3_SpiPro", "C3_InfSpiPro", 
        "C3_InfCan", "C3_PosInfFac", "C3_AntInfFac", "C3_PosMedBod", 
        "C3_PosInfBod", "C3_InfMedBod", "C2Joint", "C2RotationCenter"]],
        [["C3RotationCenter", "C4RotationCenter"],["C4_AntInfBod", "C4_AntMedBod", "C4_AntSupBod", 
        "C4_SupMedBod", "C4_PosSupBod", "C4_AntSupFac", "C4_PosSupFac", 
        "C4_SupCan", "C4_SupSpiPro", "C4_SpiPro", "C4_InfSpiPro", 
        "C4_InfCan", "C4_PosInfFac", "C4_AntInfFac", "C4_PosMedBod", 
        "C4_PosInfBod", "C4_InfMedBod", "C3Joint", "C3RotationCenter"]],
        [["C4RotationCenter", "C5RotationCenter"],["C5_AntInfBod", "C5_AntMedBod", "C5_AntSupBod", 
        "C5_SupMedBod", "C5_PosSupBod", "C5_AntSupFac", "C5_PosSupFac", 
        "C5_SupCan", "C5_SupSpiPro", "C5_SpiPro", "C5_InfSpiPro", 
        "C5_InfCan", "C5_PosInfFac", "C5_AntInfFac", "C5_PosMedBod", 
        "C5_PosInfBod", "C5_InfMedBod", "C4Joint", "C4RotationCenter"]],
        [["C5RotationCenter", "C6RotationCenter"],["C6_AntInfBod", "C6_AntMedBod", "C6_AntSupBod", 
        "C6_SupMedBod", "C6_PosSupBod", "C6_AntSupFac", "C6_PosSupFac", 
        "C6_SupCan", "C6_SupSpiPro", "C6_SpiPro", "C6_InfSpiPro", 
        "C6_InfCan", "C6_PosInfFac", "C6_AntInfFac", "C6_PosMedBod", 
        "C6_PosInfBod", "C6_InfMedBod", "C5Joint", "C5RotationCenter"]],
        [["C6RotationCenter", "C7RotationCenter"],["C7_AntInfBod", "C7_AntMedBod", "C7_AntSupBod", 
        "C7_SupMedBod", "C7_PosSupBod", "C7_AntSupFac", "C7_PosSupFac", 
        "C7_SupCan", "C7_SupSpiPro", "C7_SpiPro", "C7_InfSpiPro", 
        "C7_InfCan", "C7_PosInfFac", "C7_AntInfFac", "C7_PosMedBod", 
        "C7_PosInfBod", "C7_InfMedBod", "C6Joint", "C6RotationCenter", "C7Joint", "C7RotationCenter"]]]
        
    head_positioning_list = [["Tragion", "HeadRotationCenter"], ["Tragion", "Infraorbitale", "AntOccCon", "PosOccCon"]]
  
   
    # coordinate systems defined by anterior inferior and posterior inferior body points
    vertebra_coordinate_systems = {"C7":["C7_AntInfBod", "C7_PosInfBod"],
            "C6":["C6_AntInfBod", "C6_PosInfBod"],
            "C5":["C5_AntInfBod", "C5_PosInfBod"],
            "C4":["C4_AntInfBod", "C4_PosInfBod"],
            "C3":["C3_AntInfBod", "C3_PosInfBod"],
            "C2":["C2_AntInfBod", "C2_PosInfBod"] } # no separate definition for C1
            
    
    def __init__(self, filename=None):
        self.landmark_names = [] # stores names in the order they're loaded from a file
        self.landmarks = {}
        self.coordinates = None # stores an np.array of coordinates in order
        self.fitting_points = None # pts used to fit a spline
        
        self.articulated_landmarks = {} # landmark locations after articulation
        self.translated_landmarks = {} # after optional translation
        
        if filename:
            self.load_landmarks(filename)
            self.add_joints()
            
        if self.landmarks:
            self.spline = SpineSpline(self)
            
    # def __repr__(self):
    #     return "SpineModel2D()"
            
    def set_landmark(self, coords, name, add_to_names=True):
        """ Set an individual landmark. """
        self.landmarks[name] = coords
        if add_to_names:
            self.landmark_names.append(name)
            
    def set_landmarks(self, coords, names, replace=True, add_joints=True, add_spline=True):
        """ Set all the named landmarks, by default updating joints and spline. """
        if replace:
            self.landmarks = {}
            for n, c in zip(names, coords):
                self.set_landmark(c, n, add_to_names=True)
        else:
            for n, c in zip(names, coords):
                self.set_landmark(c, n, add_to_names=False)
         
        if add_joints:
            self.add_joints()
            
        if add_spline:
            self.spline = SpineSpline(self)
                            
    def get_coordinates(self):
        """ Return an np.array of the coordinate values in order. """
        if self.coordinates:
            return self.coordinates
            
        self.coordinates = np.array([self.landmarks[n] for n in self.landmark_names])
    
        return self.coordinates
        
    def get_spline(self, regenerate=True):
        """ Set and return a spline based on the current points. """
        self.spline = SpineSpline(self)
        
    def add_joints(self):
        """ computes joint locations based on the landmarks """
        
        for j in SpineModel2D.joint_points:
            sup, inf = [self.landmarks[n] for n in j[:2]]
            self.landmarks[j[2]] = np.mean([sup, inf], axis=0)
        
        self.landmarks["C7Joint"] = self.landmarks["C7_InfMedBod"] + 0.2*(self.landmarks["C7_InfMedBod"] -                      
                self.landmarks["C7_SupMedBod"]) 
        

        # new rotation centers
        suffixes = [ "_AntInfBod", "_PosInfBod", "_AntSupBod", "_PosSupBod", "_InfMedBod", "_AntMedBod", "_SupMedBod"]
        
        for j in self.icrFractions: 
            ant_inf, pos_inf, ant_sup, pos_sup, inf_med, ant_med, sup_med = [self.landmarks[j[1]+s] for s in suffixes]
            frac_x, frac_z = (j[2], j[3])

            # need to find an inscribed point per Amevo method
            x_vec = normalize(pos_inf - ant_inf)
            z_vec = np.array([-x_vec[1], x_vec[0]])
            rm = np.array([x_vec, z_vec]).transpose()
            
            # get x coord of ant_med
            ant_med_x = (ant_med-ant_inf).dot(rm)[0]
            inf_med_z = (inf_med-ant_inf).dot(rm)[1]
            
            #print( "ant_med_x: ", ant_med_x)
            #print( "inf_med_z: ", inf_med_z)
            
            x_len = distance(ant_inf, pos_inf) - ant_med_x 
            z_len = distance(inf_med, sup_med)
            
            self.landmarks[j[0]] = ant_inf + np.array([ant_med_x, inf_med_z]) + frac_x*x_len*x_vec + frac_z*z_len*z_vec
            
                    
        # c7 is special case
        self.landmarks["C7RotationCenter"] = self.landmarks["C7Joint"]
        
        # head is a special case
        self.landmarks["HeadRotationCenter"] = (self.landmarks["AntOccCon"] + self.landmarks["PosOccCon"])/2.
        
    def get_head_angle(self):
        """ Return angle of tragion to io vector wrt forward X axis in degrees; positive is eyes above ears (neck extension). """
        tragion = self.landmarks["Tragion"]
        io = self.landmarks["Infraorbitale"]
        io_trag = tragion-io
        ang = -1*np.arctan2(io_trag[1], io_trag[0])
        return ang*180/np.pi 
        
    def get_chord_angle(self):
        """ Return chord angle based on C7Joint to top of dens. Note that this is 180-mma definition. """
        c7joint = self.landmarks["C7Joint"]
        dens = self.landmarks["C2_SupPosDen"]
        vec = dens - c7joint
        ang = np.arctan2(vec[1], vec[0])
        return ang*180/np.pi


    def load_landmarks(self, filename):
        """ Load a landmark file. """
        
        with open(filename, 'r') as file_in:
            lines_in = file_in.readlines()
            # parse to list; landmark name is last
           
            lines_in = [a.split('\n')[0] for a in lines_in]
        
        # load landmarks (lines of length 3) ignoring other lines
        # store landmark names in order
        for line in lines_in:
            line_list = line.split()
            if len(line_list) == 3:
                self.landmarks[line_list[2]] = np.array([float(v) for v in line_list[0:2]])
                self.landmark_names.append(line_list[2])
                
        self.get_coordinates() # loads self.coordinates as an np.array

        return self.landmarks   
        
    def export_landmarks(self, filename=EXPORT_FILENAME, export_segment_locations=True):
        """ Write landmarks to the specified file. """
        land = self.landmarks
        if self.articulated_landmarks:
            land = self.articulated_landmarks
        if self.translated_landmarks:
            land = self.translated_landmarks
        i = 0
        with open(filename, 'w', newline='') as file_out:
            writer = csv.writer(file_out, delimiter="\t")
            for n in self.landmark_names:
                try:
                    line = [float('%.3f'%v) for v in land[n]]
                    line.append(n)
                    writer.writerow(line)
                    i += 1 
                except KeyError:
                    # print("Can't write ", n)
                    pass
                    
            if export_segment_locations:
                # export joints
                joint_names = ["C7Joint", "C6Joint", "C5Joint", "C4Joint", "C3Joint", "C2Joint", "C2_SupMidDen"]
                rotation_names = ["C7RotationCenter", "C6RotationCenter", "C5RotationCenter", "C4RotationCenter",
                "C3RotationCenter", "C2RotationCenter", "HeadRotationCenter"]
                
                for n in joint_names+rotation_names:
                    line = [float('%.3f'%v) for v in land[n]]
                    line.append(n)
                    writer.writerow(line)
                    i += 1
                
                levels = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7']
                seg_res = self.calculate_segment_coordinate_system(levels, return_dict=True)
                writer.writerow(['Segment', 'X', 'Z', 'Angle'])
                for seg in levels:
                    line = [seg] + ['{:.3f}'.format(v) for v in seg_res[seg][0]] + ['{:.3f}'.format(seg_res[seg][1]) ]
                    writer.writerow(line)
                
        print("Wrote ", i, " landmarks.")
        
                
    def articulate(self, delta_head_angle):
        """ Articulate the model based on the specified change in head angle. """
        chord_angle = 180 - self.get_chord_angle() # switch to definition in Mma, which is wrt negative x axis !!
        # print("current chord angle, mma def: ", chord_angle)
        
        delta_chord_angle = 0.515*delta_head_angle # MAGIC NUMBER
        new_chord_angle = chord_angle + delta_chord_angle # still in Mma definition
        starting_coeffs = self.spline.original_coefficients # these are constant
        new_coeffs = delta_chord_angle*np.array([0.78106, 0.321068]) + starting_coeffs # MAGIC NUMBERS
        
        
        new_joints = self.spline.update_spline(new_coeffs, 180 - new_chord_angle) # convert from Mma def
        
        # now set vertebra per these joints
        
        self.position_spine(new_joints)
        
        #  now set head
  
        old_rotation_center = self.landmarks["HeadRotationCenter"]
        new_rotation_center = self.articulated_landmarks["HeadRotationCenter"]
        ang_rad = -delta_head_angle*np.pi/180.
        
        rm = np.array([[np.cos(ang_rad), np.sin(ang_rad)],[-np.sin(ang_rad),np.cos(ang_rad)]]).transpose()
        
        for land in self.head_positioning_list[1]:
            self.articulated_landmarks[land] = np.dot(rm, self.landmarks[land] - old_rotation_center) + new_rotation_center
            
    def articulate_ROM(self, delta_head_angle):
        """ Articulate the model based on the specified change in head angle, using distribution of ROM from Snyder, 
          separately for flexion and extension """
        
        # get segment deltas
        # starts with head, then C2C3...

        segment_deltas = [f*delta_head_angle for f in self.romFractions[::-1]] # note reversing order to start low
        
        
        # new_joints = self.spline.update_spline(new_coeffs, 180 - new_chord_angle) # convert from Mma def
        
        joint_names = ["C7RotationCenter", "C6RotationCenter", "C5RotationCenter", "C4RotationCenter", 
            "C3RotationCenter", "C2RotationCenter", "C2_SupMidDen"]
        
        
        vpos_pairs =  [a[0] for a in self.vertebra_positioning_list][::-1] # reverse to start low
        joint_names = ["C7RotationCenter"] + [a[0] for a in vpos_pairs]  #  these are the names of the newly computed joint centers
        
        # print("joint names: ", joint_names)
        # start from joint locations in landmarks, articulate
        # print(list(zip(vpos_pairs, self.romFractions[:-1])))
        #  print("segment_deltas: ", segment_deltas)
        
        # start from joint locations in landmarks, articulate
         
        new_joints_global = [self.landmarks[joint_names[0]]] # current lowest joint
        
        previous_total = 0  # increment the total angle change as we go, degrees
        
        for vp, delta in zip(vpos_pairs, segment_deltas[:-1]): # drop head here
            p1, p2 = self.landmarks[vp[1]], self.landmarks[vp[0]] # top to bottom order in vp
            rang = (previous_total + delta)*np.pi/180.
            previous_total = previous_total + delta # accumulate total change
            rotation_matrix = np.array([[np.cos(rang), np.sin(rang)], [-np.sin(rang), np.cos(rang)]]).transpose() 
            new_joints_global.append(new_joints_global[-1] + (p2-p1).dot(rotation_matrix) )
            
        # make a dictionary with the names
        joint_dict = {}
        for j, n in zip(new_joints_global, joint_names):
            joint_dict[n] = j

        # now set vertebra per these joints
 
        self.position_spine(joint_dict)
        
  
        old_rotation_center = self.landmarks["HeadRotationCenter"]
        new_rotation_center = self.articulated_landmarks["HeadRotationCenter"]
        ang_rad = -delta_head_angle*np.pi/180.
        
        rm = np.array([[np.cos(ang_rad), np.sin(ang_rad)],[-np.sin(ang_rad),np.cos(ang_rad)]]).transpose()
        
        for land in self.head_positioning_list[1]:
            self.articulated_landmarks[land] = np.dot(rm, self.landmarks[land] - old_rotation_center) + new_rotation_center

 
        
    def position_vertebra(self, joint_list, positioning_list):
        """ Set the vertebra based on the articulated joint locations. """
        
        ref_points = positioning_list[0]
        pts_to_place = positioning_list[1]
        
        original_location = self.landmarks[ref_points[1]]
        original_vector = normalize(self.landmarks[ref_points[0]] - self.landmarks[ref_points[1]])
        
        new_location = joint_list[ref_points[1]]
        new_vector = normalize(joint_list[ref_points[0]] - joint_list[ref_points[1]])
        
        rm = np.array([original_vector, np.array([-original_vector[1], original_vector[0]])]).transpose().dot(
            np.array([new_vector, np.array([-new_vector[1], new_vector[0]])]) )
            
        for n in pts_to_place:
            self.articulated_landmarks[n] = (self.landmarks[n] - original_location).dot(rm) + new_location
            
    
    def position_spine(self, joint_list):
        """ Set all vetebrae based on joint_list. """
        for plist in self.vertebra_positioning_list:
            self.position_vertebra(joint_list, plist)
            
    def calculate_segment_coordinate_system(self, seg_name, return_dict=True):
        """ Return the origin and x-axis orientation for the specified segment(s) """
        
        seg_list = seg_name # [i for i in list(seg_name)] # make a list
        if type(seg_list) != type([]):
            seg_list = [seg_list]
            
        land = self.landmarks
        if self.articulated_landmarks:
            land = self.articulated_landmarks
        out_list = []
        try:
            for seg in seg_list:
                pts = [land[n] for n in SpineModel2D.vertebra_coordinate_systems[seg]]
                xaxis = pts[1] - pts[0]
                out_list.append([pts[0], np.arctan(xaxis[1]/xaxis[0])*180/np.pi])
        except KeyError as e:
            print("Can't calculate coordinates for ", e)
            
        if len(out_list) == 1:
            if return_dict:
                return dict(zip(seg_list, out_list))
            else:
                return out_list[0]
        else:
            if return_dict:
                return dict(zip(seg_list, out_list))
            else:
                return out_list
    
    def translate_output(self, loc): #  loc should be a vertebra level ("C7") or [origin, x-axis angle]
        """ translate the model_out to an arbitrary location & rotation, or a specified segment coordinate system """
        if type(loc) == str: 
            cres = self.calculate_segment_coordinate_system(loc, return_dict=True)
            origin, angle = cres[loc]
        else:
            origin, angle = loc
            
        print(origin, angle)
            
        ang = angle*np.pi/180.
        
        rm = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]).transpose()    
            
        land = self.landmarks
        if self.articulated_landmarks:
            land = self.articulated_landmarks
        
        for n, p in land.items():
            if type(loc) == str:
                self.translated_landmarks[n] = (p - origin).dot(rm.transpose())
            else:
                self.translated_landmarks[n] = p.dot(rm) + origin
                        
        
class anthro(object):
    """ Store anthro vars """
    FEMALE = 1
    MALE = -1
    
    def __init__(self, sex=FEMALE, stature=1627, age=20, shs=0.52):
        self.sex = sex
        self.stature = stature
        self.age = age
        self.shs = shs
        
    def vector(self):
        """Return a vector in the correct form for the regression model """
        # ideally this would read the correct form for the vector from the regression model or have it passed as an arg
        
        # return np.array([self.sex, self.stature, self.sex*self.stature, self.age, 1])
        # return np.array([self.stature, self.age, self.shs, 1])  
       
        return np.array([self.sex, self.stature, self.shs, self.age, self.sex*self.stature,
                self.sex*self.shs, self.sex*self.age, 1])   # REGRESSION_MODEL
        

class PCARSpine2D(object):
    """ Define a PCAR object for generating 2D spines """
    
    
    def __init__(self, pc_matrix_filename=PC_MATRIX_FILENAME,regression_coefficients_filename=REGRESSION_COEFFICIENTS_FILENAME,
        mean_model_filename=MEAN_MODEL_FILENAME):
        
        self.mean_model = None
        self.pcmatrix = None
        self.regression_coefficients = None
        self.model_out = None # predictions go here as SpineModel2D objects
        
        # load data from files
        
        self.load_pcmatrix(pc_matrix_filename)
        self.load_regression_coefficients(regression_coefficients_filename)
        self.load_mean_model(mean_model_filename)    
        
        
    def predict(self, anthro=None, sex=anthro.FEMALE, stature=1620, age=20, shs=0.52, export_model=True, delta_head_angle=0,
        location=None, use_rom=True):
        """ Generate a predicted c-spine based on anthro args.  
            args can be keywords: sex=-1|1, stature=mm, age=yrs
            or an anthro object
            or a list or np.array in the correct form """
            
        # anthro_vector = np.array([sex, stature, sex*stature, age, 1])
        # anthro_vector = np.array([stature, age, shs, 1])
        # {"sex", "stature", "sh/s", "age", "sex*stature", "sex*shs", "sex*age", "const"}
        anthro_vector = np.array([sex, stature, shs, age, sex*stature, sex*shs, sex*age, 1])  # REGRESSION_MODEL
        
        if anthro:
            try:
                anthro_vector = anthro.vector()
            except AttributeError:
                anthro_in = list(anthro)
                # anthro_vector = np.array([anthro_in[0], anthro_in[1], anthro_in[0]*anthro_in[1], anthro_in[2], 1])
                # modified for no-sex version: stature, age, shs
                # anthro_vector = np.array([anthro_in[0], anthro_in[1], anthro_in[2], 1]) 
                # modified for sex-interaction version 2017-02-25
               
                anthro_vector = np.array([anthro_in[0], anthro_in[1], anthro_in[2], anthro_in[3],
                        anthro_in[0]*anthro_in[1], anthro_in[0]*anthro_in[2], anthro_in[0]*anthro_in[3], 1])   # REGRESSION_MODEL
                
        print("anthro_vector: ", anthro_vector)
        
        self.scores = self.regression_coefficients.dot(anthro_vector) 
        self.centered_coords = self.pcmatrix.dot(self.scores).reshape((-1, 2))
        self.coordinates = self.mean_model.coordinates + self.centered_coords
        
        self.model_out = SpineModel2D() # create a new spine model with new points
        self.model_out.set_landmarks(self.coordinates, self.mean_model.landmark_names, replace=True)
        
        if use_rom: 
            self.model_out.articulate_ROM(delta_head_angle)
        else:
            self.model_out.articulate(delta_head_angle)
        
        if location:
            loc = location if type(location) == str else [np.array(location[0]), location[1]]
            self.model_out.translate_output(loc)
        
        if export_model:
            self.model_out.export_landmarks()
    
        
    def load_pcmatrix(self, filename):
        """ load the principal component matrix from the specified tab-delimited file """
        
        with open(filename, 'r') as file_in:
           self.pcmatrix = np.loadtxt(file_in)
        
        # transpose the pcmatrix
        self.pcmatrix = self.pcmatrix.transpose()
           
           
    def load_regression_coefficients(self, filename):
        """ load the regression coefficient  matrix from the specified tab-delimited file """
        
        with open(filename, 'r') as file_in:
           self.regression_coefficients = np.loadtxt(file_in)
        
        
    def load_mean_model(self, filename):
        """ load the mean model """
        
        self.mean_model = SpineModel2D(filename)
        

# functions for SpineSpline

def Bernstein(n, k):
    """Bernstein polynomial function"""
    coeff = binom(n, k)

    def _bpoly(x):
        if x>=1 or x<0:
            return 0.0
        else:
            return coeff * (x**k) * (1 - x)**(n - k)

    return _bpoly
    

def eval_basis_functions(basis, t):
    val = 0
    for b in basis:
        val += b(t)
        
    return val
    
    
def get_spline_function(coeffs, return_function=True):
    
    basis = [Bernstein(3, 1), Bernstein(3, 2)]    
    xvals = np.linspace(0, 1.0, 20)    
    out = []

    def _spline(t):
        val = 0
        for c, f in zip(coeffs, basis):
            val += c*f(t)
        return val
        
    if return_function:
        return _spline
    else:
    
        for x in xvals:
            out.append(_spline(x))
            
        return out
        
def normalize(v):
    return v/np.linalg.norm(v)
    
def distance(p1, p2):
    return np.linalg.norm(p2-p1)
    

def print_value_table(out_tab):
    """ print a table of lists of values """
    for line in out_tab:
        for v in line:
            print("{:.2f}\t".format(v),)
        print()
    

class SpineSpline(object):
    """ class to generate and manipulate a spline representing a spine posture """
    
    def __init__(self, spine):
        self.spine_model = spine
        self.coefficients = None
        self.xvalues = None
        self.points = None
        
        self.points = [spine.landmarks[n] for n in SpineModel2D.spline_fit_points]
            
        # pts are in top-to-bottom order
        
        
        # obtain segment lengths: reverse for bottom to top order
        self.segment_lengths = []
        last_pt = self.points[0]
        for p in self.points[1:]:
            self.segment_lengths.append(distance(last_pt, p))
            last_pt = p
        self.segment_lengths.reverse()    
        
            
        # rotate chord to x axis
        chord_vector = normalize(self.points[0] - self.points[-1])
        perp_vector = np.array([-chord_vector[1], chord_vector[0]])
        self.rotation_matrix = np.array([chord_vector, perp_vector]).transpose()
        rotated_pts = [(v-self.points[-1]).dot(self.rotation_matrix) for v in self.points]
        rotated_pts.reverse()
        
        self.coefficients, self.xvalues, self.scaled_spline_function, self.spline_function = \
                    self.fit_B_Spline(rotated_pts) # [coeffs, non-scaled xvals, scaled function, unscaled function ]
                    
        self.original_coefficients = self.coefficients.copy()
        
        self.length = self.spline_length()
        self.chord_length = rotated_pts[-1][0]
        
        #print "spline length: ", self.length
        #print "chord length: ", self.chord_length
        
    
    def get_joints(self):
        """ return the joints used to position the vertebra as a dictionary with coords in global space """
        
        joints = self.find_spline_joints() 
        rmi = self.rotation_matrix.transpose() 
        joints_global = [v.dot(rmi) + self.points[-1] for v in joints]
         
        # joint_names = ["C7Joint", "C6Joint", "C5Joint", "C4Joint", "C3Joint", "C2Joint", "C2_SupMidDen"]
        joint_names = ["C7RotationCenter", "C6RotationCenter", "C5RotationCenter", "C4RotationCenter",
                "C3RotationCenter", "C2RotationCenter", "C2_SupMidDen"]
            
        joint_dict = {}
        for j, n in zip(joints_global, joint_names):
            joint_dict[n] = j
            
            
        return joint_dict
        

        
    def fit_B_Spline(self, pts):
        """ fit a y=f(x) b-spline with two coeffs; y values should start and end at zero; x values should be zero to chord length """
        
        xvals = [p[0] for p in pts]
        yvals = [p[1] for p in pts]
        
        basis = [Bernstein(3, 1), Bernstein(3, 2)]
        
        # evaluate basis at x vals
        scaled_xvals = [x/xvals[-1] for x in xvals]
        design_matrix = []
        for x in scaled_xvals:
            design_matrix.append([basis[0](x), basis[1](x)])
            
        design_matrix = np.array(design_matrix)
        coeffs = np.linalg.pinv(design_matrix).dot(np.array(yvals)) # least-squares fit
        
        # print "coeffs: ", coeffs
        
        # get function for argument 0-1
        sfn = get_spline_function(coeffs, return_function=True)
        
        def _sfn_chord(x):
            return sfn(x/xvals[-1])
    
        # out_tab = [[x, y, sfn(x), y - sfn(x)] for x, y in zip(scaled_xvals, yvals)]
        # print_value_table(out_tab)
                        
        return [coeffs, xvals, sfn, _sfn_chord ]
        
        
    def spline_length(self, numpts=500):
        """ estimate spline curve length """
        
        fn = self.spline_function # non-scaled arg
        xvals = np.linspace(0, self.xvalues[-1], numpts)
                
        pts = [np.array([x, fn(x)]) for x in xvals]
        
        length = 0
        lastpt = pts[0]
        for p in pts[1:]:
            length += distance(lastpt, p)
            lastpt = p
            
        return length
    
    def find_spline_point(self, x, d, xstart):
        """ returns x value for a point on the spline d from the point given by x; xstart is the starting value for the search """
             
        chord_fn = self.spline_function
        def _sfn(s_x): # create a spline function that returns coordinates in chord space for argument x along chord
            return np.array([s_x, chord_fn(s_x)])
            
        start_pt = np.array(_sfn(x))
        
        def _dfn(new_x):
            return (distance(start_pt, _sfn(new_x)) - d)**2 
            
        minres = minimize_scalar(_dfn, method='bounded', bounds=[x, xstart+(xstart-x)])
            
        return minres.x
        
    def find_spline_joints(self):
        """ finds joint locations along current spline based on self.segment_lengths """
        
        joint_locations = [np.array([0,0])] # since we're in chord space, C7Joint is always at origin
        last_x = 0
        
        for sl, sp in zip(self.segment_lengths, self.xvalues[1:]): # use current spline xvals as starting points
            new_x = self.find_spline_point(last_x, sl, sp) 
            joint_locations.append(np.array([new_x, self.spline_function(new_x)]))
            last_x = new_x
            
        return joint_locations
    
    
    def update_spline(self, coeffs, chord_angle=None):
        """ updates the spline based on coeffs, holding segment lengths constant
            returns a dictionary with the new joint locations along the spline """
                
        self.coefficients = coeffs
        # get new functions and xvals
        self.scaled_spline_function = get_spline_function(self.coefficients, return_function=True) # scaled x
        
        def _sfn_chord(x): # but this is using the old max x value
            return self.scaled_spline_function(x/self.xvalues[-1])
            
        self.spline_function = _sfn_chord

        new_joints = self.find_spline_joints()
        old_max = self.xvalues[-1]
        new_max = new_joints[-1][0]
        def _sfn_chord_new(x): # but this is using the old max x value
            return self.scaled_spline_function(x/new_max)
        
        self.spline_function = _sfn_chord_new
        
        self.xvalues = [j[0] for j in new_joints]
        
        # update rotation matrix for new chord_angle
       # print "original_chord angle: ", self.spine_model.get_chord_angle()
       # print "chord angle: ", chord_angle
        rad_ang = chord_angle*np.pi/180.
       # print "rad_ang: ", rad_ang
        
        new_chord_vector = np.array([np.cos(rad_ang), np.sin(rad_ang)])
        perp_vector = np.array([-new_chord_vector[1], new_chord_vector[0]])
        new_rm = np.array([new_chord_vector, perp_vector])
       # print "new_chord_vector: ", new_chord_vector
    
        new_joints = self.find_spline_joints() # in chord coords with x along chord
        
        new_joints_global = [j.dot(new_rm) + self.points[-1] for j in new_joints]
        
        # print "njg: ", new_joints_global
        # joint_names = ["C7Joint", "C6Joint", "C5Joint", "C4Joint", "C3Joint", "C2Joint", "C2_SupMidDen"]
        joint_names = ["C7RotationCenter", "C6RotationCenter", "C5RotationCenter", "C4RotationCenter", 
            "C3RotationCenter", "C2RotationCenter", "C2_SupMidDen"]
        
        joint_dict = {}
        for j, n in zip(new_joints_global, joint_names):
            joint_dict[n] = j
            
        return joint_dict
               

if __name__ == '__main__': # when run as module

    pcar = PCARSpine2D()
    if len(sys.argv) > 1:  # interpret arguments as anthro
        num_args = [float(a) for a in sys.argv[1:]]
        target_anthro = anthro(sex=num_args[0], stature=num_args[1], shs=num_args[2], age=num_args[3] )
        pcar.predict(target_anthro, delta_head_angle=num_args[4])
    else:
        target_anthro = anthro(anthro.FEMALE, stature=1626, shs=0.52, age=27 )  # midsize Army female
        
        # uncomment this line for manual input
        target_anthro = anthro(anthro.FEMALE, stature=1626, shs=0.52, age=27 )  # build model
        
        print("Predicting...")
        pcar.predict(target_anthro, delta_head_angle=0, use_rom=True) # neutral posture
        
#       example using translation and rotation to reposition the spine
#        pcar.predict(target_anthro, delta_head_angle=0, location=[[40, 40], 30]) # neutral posture
        
        # origin is T1 anterior superior, X-axis passes through posterior T1 process landmark

    # also do 3D bone mapping
    from CSpine3DFitting import *
    bm = BoneMapper()

 
 
 
 
 
 