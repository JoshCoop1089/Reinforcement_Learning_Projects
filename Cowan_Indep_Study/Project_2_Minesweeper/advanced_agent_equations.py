
import pprint
import basic_agent as ba
    # get_indiv_cell_info -> gets everything for that cell 
    # build_fact_dict -> keeps track of known mines/safes/unkown neighbors  
    # find_num_unknowns_on_board -> # unkowns left to explore
import base_functions as bf
import advanced_agent_constraints as adv



"""
Advanced agent using equations logic inferences 

1. Find intersection between two neighbor mine lists
2. Create list1 w/o intersection and list2 w/o intersection
3. Take the longer list (either 1 or 2) and subtract the other list from it
4. Subtract the num_mine_neighbors from each other
5. Special cases: If num2-num1 == len(list2 w/o intersection) then list1 w/o intersection is totally safe, and list2 w/o intersection is totally mined
6. Add new facts to the known mine/known safes list, and update the base facts dict (see Part3 5a/5b)

"""

def apply_equations(c_board, ref_board):
    fact_dict = ba.build_fact_dictionary(c_board, ref_board)
    eq_sorted = [] # [[list, num], [list, num]]
    
    #V = [status, num_possible_mines, known_mines, hidden_neighbors, safe_neighbors]
    for v in fact_dict.values():
        remaining_mines = v[1] - len(v[2]) #number of mines - known mines
        eq_sorted.append([v[3], remaining_mines])
    
    #sort by length of neighbors 
   # eq_sorted = sorted(eq_sorted, key = lambda x: len(x[0]))
    # print("equations after sorting:")
    # pprint.pprint(eq_sorted)
    equations_after = []
    # infer_mines = set()
    # infer_safes = set()           
    # done = [] 
    # changes_made = False
    
    for eq1 in eq_sorted:
        for eq2 in eq_sorted:
            
            #same equation or one doesnt exist
            if eq1[0] == eq2[0] or not eq2 or not eq1:
                continue
            
            #write as sets
            neigh1 = set(eq1[0])
            neigh2 = set(eq2[0])
            
            # print("making equations out of: ", neigh1, " and :", neigh2)
            
            #if they cant be combined, add them as seperate equations
            # if neigh1.isdisjoint(neigh2) or neigh2.isdisjoint(neigh1): #might need to edit this
            #     # print("disjoint ")
            #     if neigh1 not in equations_after[0][1] or neigh2 not in equations_after[0][2]:  
            #         equations_after.append([[[], [], neigh1], eq1[1]])
            #         equations_after.append([[[], [], neigh2], eq2[1]])
            #     continue
            
                
            if neigh1.issubset(neigh2):
                # print("l1 < l2 ")
                total = neigh2 - neigh1
                # new_neigh1 = neigh1.difference(neigh2)
                # new_neigh2 = neigh2.difference(neigh1)
                num = eq2[1] - eq1[1]
                if num > len(total):
                    continue
                if [total, num] not in equations_after:
                    curr = [total, num]
                    equations_after.append(curr)
                else:
                    continue    
                # continue 
            
           # if len(neigh2) < len(neigh1):
            if neigh2.issubset(neigh1):
                # print("l2 < l1")
                total = neigh1 - neigh2
                # new_neigh1 = neigh1.difference(neigh2)
                # new_neigh2 = neigh2.difference(neigh1)
                num = eq1[1] - eq2[1]
                if num > len(total):
                    continue
                if [total, num] not in equations_after:
                    curr = [total, num]
                    equations_after.append(curr)
    
    return equations_after 


    
def update_equations(equations, fact_dict, cb_copy, r_board ,ks, km):
    # print("---UPDATING, EQ BEFORE: ---")
    # print(equations)
    update = False
    if len(equations) !=0:
        for [eq,n] in equations:
            for cell in eq:
                if cell in ks:
                    eq.remove(cell)
                    update = True
                
                if cell in km: #decrease number of unknown mines
                    eq.remove(cell)
                    n -= 1
                    update = True
    
#    print("---UPDATING, EQ AFTER: ---")
    # print(equations, "update = ", update)
    
    return equations, update        
 
#checks for mines or opens
def infer_from_equations(equation, fact_dict, c_board, r_board, known_safes, known_mines):
    #V = [status, num_possible_mines, known_mines, hidden_neighbors, safe_neighbors]
    # known_mines = set()
    # known_safes = set()
    eq = equation[0]
    n = equation[1]
    # print("------INFERRING--------")
    changed = False
    #equations_copy = equations.copy()
    inferred = []
    # eq = [[new_neigh1, new_neigh2, total], num]
    # print("inferring:", equation)
   # if eq[2] not in inferred:   
    if (len(eq) != 0 ) and n >=0: 

        #all neighbors in equation are safe    
        if n == 0:
            print("all are safe!", eq)
            # changed = True
            for cell in eq:
                if cell not in known_safes:
                    v = r_board[cell[0]][cell[1]] #the actual value 
                    if v != "M":
                        changed = True
                        c_board[cell[0]][cell[1]] = v #update it to show on the board
                        known_safes.add(cell)
                    else:
                        print("wrong, ", cell, "isnt safe")   
            inferred.append(eq)
                        
       
        #all cells are mines
        if n == len(eq):
            print("al are mines!", eq)
            # changed = True
            for cell in eq:
                if cell not in known_mines:
                    v = r_board[cell[0]][cell[1]] #mine value 
                    if v == "M":
                        changed = True
                        c_board[cell[0]][cell[1]] = v #update it to show on the board
                        known_mines.add(cell)
                    else:
                        print("wrong, ", cell, "isnt a mine")   

            inferred.append(eq)
   

    # print("------DONE--------")                
    return changed, known_safes, known_mines, inferred, c_board
 
 
def update_facts(ks, km, c_board, r_board):
    if len(ks) != 0 or len(km) != 0:
        #changes_made = True
        # print(f"***New info found by Constraint Satisfaction from {most_freq_unknown}***")
        # print(f"{len(new_KM)} new mines, {len(new_KS)} new safe spots")
        for (x,y) in km:
            val = r_board[x][y]
            if val == "M":
                c_board[x][y] = val
            # else:
            #     print(f"Incorrect Mine Predicted at {(x,y)}")
        for (x,y) in ks:
            val = r_board[x][y]
            if val == "?":
                c_board[x][y] = val
            # else:
            #     print(f"Incorrect Safe Predicted at {(x,y)}")   
    
    return c_board


"""
1. uncover location randomly
2. if its a mine -> minus total score
3. if not -> dont change the score 
4. add current state of the board to fact dict (build fact dict) 
        calculate equations
        infer from equations
        get the facts after first equations go through
        if you got new info
            reupdate the equations based on the fact dict 
                if in mine -> decrement n, safe -> just remove it
            call the inference function to see if u can do anything with new equations 
            repeat until 
                cant make anything more out of given info, discovered everything
                or unknowns finished

 
5. repeat the loop until all cells on the board are explored
"""               
def run_advanced_equations(c_board, r_board, num_mines):

    total_score = num_mines #decrease when the cell we land on is a mine
    random_guess = 0 #to compare against the other agents  
    unknowns = ba.find_num_unknowns_on_board(c_board) #all ? 
    cb_copy = c_board.copy()
    ks = set()
    km = set()
    # inferred = []
    equations_made = []    
    #call inferences 
    fact_dict = {}
    while unknowns > 0:
        #uncover a random spot after inferring everything you can
        #you can do anything the 1st time so it picks one randomly at start
        # print(f"\n--> Random Guess: Round {random_guess}")
        cb_copy, boom = ba.uncover_random_spot(cb_copy, r_board)
        random_guess += 1
        if boom: #mine chosen was a bomb
            total_score -= 1
        # bf.print_board(cb_copy) #see which cell was uncovered
        fact_dict = ba.build_fact_dictionary(cb_copy, r_board)  
        # pprint.pprint(fact_dict)
        cb_copy = ba.apply_logic_to_fact_dict(cb_copy, r_board, fact_dict)
        fact_dict = ba.build_fact_dictionary(cb_copy, r_board)
        # print("----SINGLE CELL INFER-----")
        # bf.print_board(cb_copy) #see which cell was uncovered
        # pprint.pprint(fact_dict)
        while True:
            #try inferring with the current state of board
            try:
                # print("----------------EQUATIONS STUFF-----------------")
                equations_made = apply_equations(cb_copy, r_board)
                if len(equations_made) != 0:
                    #infer from equations and update the board
                    while True: 
                        for eq in equations_made:
                            updated, ks, km, inferred_new, cb_copy = infer_from_equations(eq, fact_dict, cb_copy, r_board, ks, km)
                            
                            if updated: 
                                # print("updated?")
                                equations_made.remove(eq)
                                #get the fact dict after equations once
                                cb_copy = update_facts(ks, km, cb_copy, r_board)
                                # print("------changed board?-------")
                                # bf.print_board(cb_copy)
                                fact_dict = ba.build_fact_dictionary(cb_copy, r_board)
                                # equations_made, changed = update_equations(equations_made, fact_dict, cb_copy, r_board, ks, km)
                                # print("equations adjustedddd!!!")
                                # pprint.pprint(equations_made) 
                                # if not changed:    
                                #     break
                            
                        
                        equations_updated, changed = update_equations(equations_made, fact_dict, cb_copy, r_board, ks, km)
                        
                        if equations_updated == equations_made or not changed:
                            break
                       
                            # print("equations adjustedddd!!!")               
                            # pprint.pprint(equations_made)                                        
                else:
                    break
                # print("board from equations")    
                cb_copy = update_facts(ks, km, cb_copy, r_board)                    
                # bf.print_board(cb_copy)
                #print("figure out anything new? ", new_info)             
               
                fact_dict = ba.build_fact_dictionary(cb_copy, r_board)
                # print("new fact dict after equations: ")
                # pprint.pprint(fact_dict)
                cb_copy = ba.apply_logic_to_fact_dict(cb_copy, r_board, fact_dict)
                break
            except Exception:
                break
            
        unknowns = ba.find_num_unknowns_on_board(cb_copy)
    print("-"*5 + "Board Solved"+ "-"*5)   
    print(f"Final Score for Equations: {total_score} out of {num_mines} safely found!")
    print(f"Random Moves Needed Equations: {random_guess}")
    return total_score, random_guess


dimension = 5
num_mines = 5
cover, rb, mloc = bf.make_board(dimension, num_mines) 
cover1 = cover.copy() 
cover2 = cover.copy() 
cover3 = cover.copy() 


score_b, count_b = ba.run_basic_agent(cover2, rb, num_mines)
score_a, count_a = adv.run_advanced_agent(cover3, rb, num_mines)
score, count = run_advanced_equations(cover1, rb, num_mines)
bf.print_board(rb)
# bf.print_board(cover)