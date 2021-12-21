# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 21:03:31 2021

@author: joshc
"""
import base_functions as bf
import basic_agent as ba
import adv_agent_csp_v3 as adv3

covered = [["?", "?", "2"],
           ["?", "?", "?"],
           ["?", "3", "?"]]
reference = [["1", "M", "2"],
             ["2", "4", "M"],
             ["M", "3", "M"]]
# bf.print_board(covered)
# cbA, change = adv.assume_a_single_square(covered, reference)
# tsb, c = ba.run_basic_agent(covered, reference, 4)

# ts, rg = adv3.run_advanced_agent(covered, reference, 4)

cb, rb, mineloc = bf.make_board(10, 40)

ts, rg = adv3.run_advanced_agent(cb,rb,40)

