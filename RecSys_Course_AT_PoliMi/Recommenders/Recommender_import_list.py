#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/04/19

@author: Maurizio Ferrari Dacrema
"""


######################################################################
##########                                                  ##########
##########                  NON PERSONALIZED                ##########
##########                                                  ##########
######################################################################
from DressipiChallenge.Recommenders.NonPersonalizedRecommender import TopPop, Random, GlobalEffects



######################################################################
##########                                                  ##########
##########                  PURE COLLABORATIVE              ##########
##########                                                  ##########
######################################################################
from DressipiChallenge.Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from DressipiChallenge.Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from DressipiChallenge.Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from DressipiChallenge.Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_SLIMElasticNetRecommender
from DressipiChallenge.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from DressipiChallenge.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from DressipiChallenge.Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from DressipiChallenge.Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from DressipiChallenge.Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from DressipiChallenge.Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender
from DressipiChallenge.Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from DressipiChallenge.Recommenders.FactorizationMachines.LightFMRecommender import LightFMCFRecommender
from DressipiChallenge.Recommenders.Neural.MultVAERecommender import MultVAERecommender_OptimizerMask as MultVAERecommender


######################################################################
##########                                                  ##########
##########                  PURE CONTENT BASED              ##########
##########                                                  ##########
######################################################################
from DressipiChallenge.Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from DressipiChallenge.Recommenders.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender



######################################################################
##########                                                  ##########
##########                       HYBRID                     ##########
##########                                                  ##########
######################################################################
from DressipiChallenge.Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from DressipiChallenge.Recommenders.KNN.UserKNN_CFCBF_Hybrid_Recommender import UserKNN_CFCBF_Hybrid_Recommender
from DressipiChallenge.Recommenders.FactorizationMachines.LightFMRecommender import LightFMUserHybridRecommender, LightFMItemHybridRecommender
