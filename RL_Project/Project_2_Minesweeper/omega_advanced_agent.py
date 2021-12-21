import pprint, copy
import time
import base_functions as bf
import basic_agent as ba
import advanced_agent_equations as equ
import advanced_agent_constraints as csp
import adv_agent_csp_v3 as adv

'''
Advanced Agent using a combined approach of the Constraint Satisfaction
Agent and the Rules of Inference Agent, as well as the basic agent.
1.) Uncover a random square on the board to begin the game
2.) Start going through the board via continued random guessing if the initial
    states don't provide enough information and then move into applying basic
    agent logic once enough information has been obtained.
3.) Once the basic agent has run its course, apply the Rules of Inferences approach and see if 
    any new information can be obtained.
    a.) If new information is obtained and the Rules of Inferences agent has run its course,
        go back to step 1 and go through the process of guessing and the basic agent again
    b.) If new information is not obtained, proceed ahead to step 4, the CSP approach
4.) Let the CSP agent run its course on the board. 
5.) Repeat these processes until all unknowns are explored
'''


def run_combined_agent(covered, reference, num_mines):

    total_score = num_mines 
    random_guess = 0 
    unknown = ba.find_num_unknowns_on_board(covered)
    c_copy = copy.deepcopy(covered)
    ks = set()
    km = set()
    equations_made = []    
    fact_dict = {}
    while unknown > 0:
        equationsChanged = False
        c_copy, detonate = ba.uncover_random_spot(c_copy, reference)
        random_guess += 1
        if detonate: 
            total_score -= 1
        fact_dict = ba.build_fact_dictionary(c_copy, reference)  
        c_copy = ba.apply_logic_to_fact_dict(c_copy, reference, fact_dict)
        fact_dict = ba.build_fact_dictionary(c_copy, reference)

        unknown = ba.find_num_unknowns_on_board(c_copy)
        if unknown == 0:
            break


        while True: #first start off by applying rules of inference method
            try:
                equations_made = equ.apply_equations(c_copy, reference)
                if len(equations_made) != 0:
                    while True: 
                        for eq in equations_made:
                            updated, ks, km, inferred_new, c_copy = equ.infer_from_equations(eq, fact_dict, c_copy, reference, ks, km)
                            if updated: 
                                equationsChanged = True
                                equations_made.remove(eq)

                                c_copy = equ.update_facts(ks, km, c_copy, reference)
                                fact_dict = ba.build_fact_dictionary(c_copy, reference)

                        equations_updated, changed = equ.update_equations(equations_made, fact_dict, c_copy, reference, ks, km)

                        if equations_updated == equations_made or not changed:
                            break 
                          
                else:
                    break 
                
                c_copy = equ.update_facts(ks, km, c_copy, reference)                            
                fact_dict = ba.build_fact_dictionary(c_copy, reference)
                c_copy = ba.apply_logic_to_fact_dict(c_copy, reference, fact_dict)
                break
            except Exception:
                break
        

        #once method is exhausted, test out the csp method
        unknown = ba.find_num_unknowns_on_board(c_copy)
        if unknown == 0:
            break
        
        if equationsChanged == True:
            continue


        while True:
            try:

                fact_dict = adv.build_fact_dictionary(c_copy, reference)

                c_copy, fact_dict, logic_changes_made = adv.apply_logic_to_fact_dict(c_copy, reference, fact_dict)

                c_copy, assume_changes_made, fact_dict = adv.assume_a_single_square(c_copy, reference, fact_dict)

                if not logic_changes_made and not assume_changes_made:
                    break
            except Exception:
                break
        

        unknown = ba.find_num_unknowns_on_board(c_copy)
    '''
    print("-"*5 + "Combined Board Solved"+ "-"*5)
    print(f"Final Score Combined: {total_score} out of {num_mines} safely found!")
    print(f"Random Moves Needed Combined: {random_guess}")'''
    return total_score, random_guess


def run_combined_agent_advanced(covered, reference, num_mines, globalInfo = False, advanced = False, rando = False):

    total_score = num_mines 
    random_guess = 0 
    unknown = ba.find_num_unknowns_on_board(covered)
    c_copy = copy.deepcopy(covered)
    ks = set()
    km = set()
    equations_made = []    
    fact_dict = {}
    while unknown > 0:
        equationsChanged = False
        if advanced:
            c_copy, detonate = adv.advanced_location_selection(c_copy, reference, fact_dict)
        else:
            c_copy, detonate = ba.uncover_random_spot(c_copy, reference)
        random_guess += 1
        if detonate: 
            total_score -= 1
        fact_dict = ba.build_fact_dictionary(c_copy, reference)  
        c_copy = ba.apply_logic_to_fact_dict(c_copy, reference, fact_dict)
        fact_dict = ba.build_fact_dictionary(c_copy, reference)
        unknown = ba.find_num_unknowns_on_board(c_copy)
        if unknown == 0:
            break

        while True: #first start off by applying rules of inference method
            try:
                equations_made = equ.apply_equations(c_copy, reference)
                if len(equations_made) != 0:
                    while True: 
                        for eq in equations_made:
                            updated, ks, km, inferred_new, c_copy = equ.infer_from_equations(eq, fact_dict, c_copy, reference, ks, km)
                            
                            if updated: 
                                equationsChanged = True
                                equations_made.remove(eq)
                                c_copy = equ.update_facts(ks, km, c_copy, reference)
                                fact_dict = ba.build_fact_dictionary(c_copy, reference)
                            
                                    
                        
                        equations_updated, changed = equ.update_equations(equations_made, fact_dict, c_copy, reference, ks, km)

                        if equations_updated == equations_made or not changed:
                            break 
                                                      
                else:
                    break 

                c_copy = equ.update_facts(ks, km, c_copy, reference)                            
                fact_dict = ba.build_fact_dictionary(c_copy, reference)
                c_copy = ba.apply_logic_to_fact_dict(c_copy, reference, fact_dict)
                break
            except Exception:
                break

        #once method is exhausted, test out the csp method
        unknown = ba.find_num_unknowns_on_board(c_copy)
        if unknown == 0:
            break
        
        if equationsChanged == True:
            continue


        while True:
            try:
                fact_dict = adv.build_fact_dictionary(c_copy, reference, globalInfo)
                c_copy, fact_dict, logic_changes_made = adv.apply_logic_to_fact_dict(c_copy, reference, fact_dict, globalInfo)
                c_copy, assume_changes_made, fact_dict = adv.assume_a_single_square(c_copy, reference, fact_dict, globalInfo)
                if not logic_changes_made and not assume_changes_made:
                    break
            except Exception:
                break
        unknown = ba.find_num_unknowns_on_board(c_copy)
    '''
    print("-"*5 + "Combined2 Board Solved"+ "-"*5)
    print(f"Final Score Combined2: {total_score} out of {num_mines} safely found!")
    print(f"Random Moves Needed Combined2: {random_guess}")'''

    return total_score, random_guess

'''
dimension = 20
num_mines = 100
cover, rb, mloc = bf.make_board(dimension, num_mines) 
cover1 = copy.deepcopy(cover) 
cover2 = copy.deepcopy(cover) 
cover3 = copy.deepcopy(cover) 
cover4 = copy.deepcopy(cover)
cover5 = copy.deepcopy(cover)


start_basic = time.time()
score_a, count_a = ba.run_basic_agent(cover1, rb, num_mines)
end_basic = time.time()

start_c = time.time()
score_d, count_d = run_combined_agent(cover4, rb, num_mines)
end_c = time.time()

start_csp = time.time()
score_b, count_b = adv.run_advanced_agent(cover2, rb, num_mines)
end_csp = time.time()

start_eq = time.time()
score_c, count_c = equ.run_advanced_equations(cover3, rb, num_mines)
end_eq = time.time()

start_c2 = time.time()
score_e, count_e = run_combined_agent_advanced(cover5, rb, num_mines)
end_c2 = time.time()

print("Basic time:", end_basic-start_basic)
print("CSP time:", end_csp-start_csp)
print("Eq time:", end_eq-start_eq)
print("Combined time:", end_c-start_c)
print("Combined2 time", end_c2-start_c2)


bf.print_board(rb)'''
