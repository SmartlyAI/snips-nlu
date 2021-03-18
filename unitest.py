#!/usr/bin/python3
# coding: utf-8 

from __future__ import unicode_literals

"""
    Smartly - Digital et Data : Snips Train Service (REST API)
    ------------------------------------------------------
    
    This module test all functionalities of Snips API.
    
    :author: MTE
    :copyright: © 2020 by Smartly and OBS D&D
    :license: Smartly, all rights reserved
"""


__version__ = "0.1"


import unittest
from pathlib import Path
from flask import Response
from flask_restful import Api
import json


class SnipsTrainTestCase(unittest.TestCase):
    """ TestCase for Snips train service module: operations on data """

    def test_a_train_data(self):
        """Test train data """


        self.assertEqual(type(result_resource), dict)
        self.assertEqual(type(result_clients), dict)
        self.assertEqual(result_resource, current_resource)
        self.assertEqual(result_clients, current_client)        
    
    def test_b_status_train(self):
        """Test train status """
        
        usr_data = {
            "nom": "Jules",
            "prenom": "Teukeu",
            "fonction": "Directeur des enseignements",
            "anciennete": 3,
            "mise_a_jour": "2020-03-06 11:55:11.827000",
            "conge": 5,
            "actif": False,
            "actionnaire": True,
            "missions": ["Bruxelle", "Paris"]
        }
        
        
        self.assertEqual(type(result_usr), tuple)
        self.assertEqual(type(result_usr[0]), dict)
        self.assertEqual(type(result_usr[1]), dict)
        self.assertNotEqual(len(result_usr[1]), len(current_clients))

    def test_c_error_data_input(self):
        """Test update files database component by id """

        self.assertEqual(type(result_resr), tuple)
        self.assertEqual(type(result_resr[0]), dict)
        self.assertEqual(type(result_resr[1]), dict)
        self.assertEqual(result_resr[0]['ntealan'], False)


    
    def test_d_error_train_status(self):
        """Test error when get train status"""

        self.assertEqual(type(result_usr), tuple)
        self.assertEqual(type(result_usr[0]), dict)
        self.assertEqual(type(result_usr[1]), dict)
        self.assertNotEqual(len(result_usr[1]), len(current_clients))



class SnipsParseTestCase(unittest.TestCase):
    """ TestCase for Snips parse service module: operations on data """

    def test_a_parse_data(self): pass

    def test_b_error_parse_data(self): pass





if __name__ == '__main__':
    unittest.main(verbosity=8)
