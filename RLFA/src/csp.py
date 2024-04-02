"""CSP (Constraint Satisfaction Problems) problems and solvers. (Chapter 6)"""

import itertools
import random
import re
import string
import time
from collections import defaultdict, Counter
from functools import reduce
from operator import eq, neg

from sortedcontainers import SortedSet

import search
from utils import argmin_random_tie, count, first, extend

#I added this global variables to help with the assignemt with the different methods 

counter=0 #global variable used for measurements of the methods

counterLimit=0 #global variable used to limit the measurements of the methods

assignmCounter=0 #global variable that counts how many nodes we visit in the search tree

class CSP(search.Problem):
    """This class describes finite-domain Constraint Satisfaction Problems.
    A CSP is specified by the following inputs:
        variables   A list of variables; each is atomic (e.g. int or string).
        domains     A dict of {var:[possible_value, ...]} entries.
        neighbors   A dict of {var:[var,...]} that for each variable lists
                    the other variables that participate in constraints.
        constraints A function f(A, a, B, b) that returns true if neighbors
                    A, B satisfy the constraint when they have values A=a, B=b
    In the textbook and in most mathematical definitions, the
    constraints are specified as explicit pairs of allowable values,
    but the formulation here is easier to express and more compact for
    most cases (for example, the n-Queens problem can be represented
    in O(n) space using this notation, instead of O(n^4) for the
    explicit representation). In terms of describing the CSP as a
    problem, that's all there is.
    However, the class also supports data structures and methods that help you
    solve CSPs by calling a search function on the CSP. Methods and slots are
    as follows, where the argument 'a' represents an assignment, which is a
    dict of {var:val} entries:
        assign(var, val, a)     Assign a[var] = val; do other bookkeeping
        unassign(var, a)        Do del a[var], plus other bookkeeping
        nconflicts(var, val, a) Return the number of other variables that
                                conflict with var=val
        curr_domains[var]       Slot: remaining consistent values for var
                                Used by constraint propagation routines.
    The following methods are used only by graph_search and tree_search:
        actions(state)          Return a list of actions
        result(state, action)   Return a successor of state
        goal_test(state)        Return true if all constraints satisfied
    The following are just for debugging purposes:
        nassigns                Slot: tracks the number of assignments made
        display(a)              Print a human-readable representation
    """

    def __init__(self, variables, domains, neighbors, constraints):
        """Construct a CSP problem. If variables is empty, it becomes domains.keys()."""
        variables = variables or list(domains.keys())
        self.variables = variables
        self.domains = domains
        self.neighbors = neighbors
        self.constraints = constraints
        self.curr_domains = domains
        self.nassigns = 0

    def assign(self, var, val, assignment):
        """Add {var: val} to assignment; Discard the old value if any."""
        assignment[var] = val
        self.nassigns += 1

    def unassign(self, var, assignment):
        """Remove {var: val} from assignment.
        DO NOT call this if you are changing a variable to a new value;
        just call assign for that."""
        if var in assignment:
            del assignment[var]

    def nconflicts(self, var, val, assignment):
        """Return the number of conflicts var=val has with other variables."""

        # Subclasses may implement this more efficiently
        def conflict(var2):
            return var2 in assignment and not self.constraints(var, val, var2, assignment[var2])

        return count(conflict(v) for v in self.neighbors[var])

    def display(self, assignment):
        """Show a human-readable representation of the CSP."""
        # Subclasses can print in a prettier way, or display with a GUI
        print(assignment)

    # These methods are for the tree and graph-search interface:

    def actions(self, state):
        """Return a list of applicable actions: non conflicting
        assignments to an unassigned variable."""
        if len(state) == len(self.variables):
            return []
        else:
            assignment = dict(state)
            var = first([v for v in self.variables if v not in assignment])
            return [(var, val) for val in self.domains[var]
                    if self.nconflicts(var, val, assignment) == 0]

    def result(self, state, action):
        """Perform an action and return the new state."""
        (var, val) = action
        return state + ((var, val),)

    def goal_test(self, state):
        """The goal is to assign all variables, with all constraints satisfied."""
        assignment = dict(state)
        return (len(assignment) == len(self.variables)
                and all(self.nconflicts(variables, assignment[variables], assignment) == 0
                        for variables in self.variables))

    # These are for constraint propagation

    def support_pruning(self):
        """Make sure we can prune values from domains. (We want to pay
        for this only if we use it.)"""
        if self.curr_domains is None:
            self.curr_domains = {v: list(self.domains[v]) for v in self.variables}

    def suppose(self, var, value):
        """Start accumulating inferences from assuming var=value."""
        self.support_pruning()
        removals = [(var, a) for a in self.curr_domains[var] if a != value]
        self.curr_domains[var] = [value]
        return removals

    def prune(self, var, value, removals):
        """Rule out var=value."""
        self.curr_domains[var].remove(value)
        if removals is not None:
            removals.append((var, value))

    def choices(self, var):
        """Return all values for var that aren't currently ruled out."""
        return (self.curr_domains or self.domains)[var]

    def infer_assignment(self):
        """Return the partial assignment implied by the current inferences."""
        self.support_pruning()
        return {v: self.curr_domains[v][0]
                for v in self.variables if 1 == len(self.curr_domains[v])}

    def restore(self, removals):
        """Undo a supposition and all inferences from it."""
        for B, b in removals:
            self.curr_domains[B].append(b)

    # This is for min_conflicts search

    def conflicted_vars(self, current):
        """Return a list of variables in current assignment that are in conflict"""
        return [var for var in self.variables
                if self.nconflicts(var, current[var], current) > 0]

# ______________________________________________________________________________
# Constraint Propagation with AC3


def no_arc_heuristic(csp, queue):
    return queue


def dom_j_up(csp, queue):
    return SortedSet(queue, key=lambda t: neg(len(csp.curr_domains[t[1]])))


def AC3(csp, queue=None, removals=None, arc_heuristic=dom_j_up):
    """[Figure 6.3]"""
    if queue is None:
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    checks = 0
    while queue:
        (Xi, Xj) = queue.pop()
        revised, checks = revise(csp, Xi, Xj, removals, checks)
        if revised:
            if not csp.curr_domains[Xi]:
                return False, checks  # CSP is inconsistent
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:
                    queue.add((Xk, Xi))
    return True, checks  # CSP is satisfiable


def revise(csp, Xi, Xj, removals, checks=0):
    """Return true if we remove a value."""
    revised = False
    for x in csp.curr_domains[Xi][:]:
        # If Xi=x conflicts with Xj=y for every possible y, eliminate Xi=x
        # if all(not csp.constraints(Xi, x, Xj, y) for y in csp.curr_domains[Xj]):
        conflict = True
        for y in csp.curr_domains[Xj]:
            if csp.constraints(Xi, x, Xj, y):
                conflict = False
            checks += 1
            if not conflict:
                break
        if conflict:
            csp.prune(Xi, x, removals)
            revised = True
    return revised, checks


# Constraint Propagation with AC3b: an improved version
# of AC3 with double-support domain-heuristic

def AC3b(csp, queue=None, removals=None, arc_heuristic=dom_j_up):
    global counter
    global counterLimit
    if queue is None:
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    checks = 0
    while queue:
        if counter > counterLimit: #This is the limit of the method measurements
           return -1 
        (Xi, Xj) = queue.pop()
        # Si_p values are all known to be supported by Xj
        # Sj_p values are all known to be supported by Xi
        # Dj - Sj_p = Sj_u values are unknown, as yet, to be supported by Xi
        Si_p, Sj_p, Sj_u, checks = partition(csp, Xi, Xj, checks)
        if not Si_p:
            return False, checks  # CSP is inconsistent
        revised = False
        for x in set(csp.curr_domains[Xi]) - Si_p:
            csp.prune(Xi, x, removals)
            revised = True
        if revised:
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:
                    queue.add((Xk, Xi))
        if (Xj, Xi) in queue:
            if isinstance(queue, set):
                # or queue -= {(Xj, Xi)} or queue.remove((Xj, Xi))
                queue.difference_update({(Xj, Xi)})
            else:
                queue.difference_update((Xj, Xi))
            # the elements in D_j which are supported by Xi are given by the union of Sj_p with the set of those
            # elements of Sj_u which further processing will show to be supported by some vi_p in Si_p
            for vj_p in Sj_u:
                for vi_p in Si_p:
                    conflict = True
                    counter+=1 #This is the counter of the measurements
                    if csp.constraints(Xj, vj_p, Xi, vi_p):
                        conflict = False
                        Sj_p.add(vj_p)
                    checks += 1
                    if not conflict:
                        break
            revised = False
            for x in set(csp.curr_domains[Xj]) - Sj_p:
                csp.prune(Xj, x, removals)
                revised = True
            if revised:
                for Xk in csp.neighbors[Xj]:
                    if Xk != Xi:
                        queue.add((Xk, Xj))
    return True, checks  # CSP is satisfiable


def partition(csp, Xi, Xj, checks=0):
    Si_p = set()
    Sj_p = set()
    Sj_u = set(csp.curr_domains[Xj])
    for vi_u in csp.curr_domains[Xi]:
        conflict = True
        # now, in order to establish support for a value vi_u in Di it seems better to try to find a support among
        # the values in Sj_u first, because for each vj_u in Sj_u the check (vi_u, vj_u) is a double-support check
        # and it is just as likely that any vj_u in Sj_u supports vi_u than it is that any vj_p in Sj_p does...
        for vj_u in Sj_u - Sj_p:
            # double-support check
            if csp.constraints(Xi, vi_u, Xj, vj_u):
                conflict = False
                Si_p.add(vi_u)
                Sj_p.add(vj_u)
            checks += 1
            if not conflict:
                break
        # ... and only if no support can be found among the elements in Sj_u, should the elements vj_p in Sj_p be used
        # for single-support checks (vi_u, vj_p)
        if conflict:
            for vj_p in Sj_p:
                # single-support check
                if csp.constraints(Xi, vi_u, Xj, vj_p):
                    conflict = False
                    Si_p.add(vi_u)
                checks += 1
                if not conflict:
                    break
    return Si_p, Sj_p, Sj_u - Sj_p, checks


# Constraint Propagation with AC4

def AC4(csp, queue=None, removals=None, arc_heuristic=dom_j_up):
    if queue is None:
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    support_counter = Counter()
    variable_value_pairs_supported = defaultdict(set)
    unsupported_variable_value_pairs = []
    checks = 0
    # construction and initialization of support sets
    while queue:
        (Xi, Xj) = queue.pop()
        revised = False
        for x in csp.curr_domains[Xi][:]:
            for y in csp.curr_domains[Xj]:
                if csp.constraints(Xi, x, Xj, y):
                    support_counter[(Xi, x, Xj)] += 1
                    variable_value_pairs_supported[(Xj, y)].add((Xi, x))
                checks += 1
            if support_counter[(Xi, x, Xj)] == 0:
                csp.prune(Xi, x, removals)
                revised = True
                unsupported_variable_value_pairs.append((Xi, x))
        if revised:
            if not csp.curr_domains[Xi]:
                return False, checks  # CSP is inconsistent
    # propagation of removed values
    while unsupported_variable_value_pairs:
        Xj, y = unsupported_variable_value_pairs.pop()
        for Xi, x in variable_value_pairs_supported[(Xj, y)]:
            revised = False
            if x in csp.curr_domains[Xi][:]:
                support_counter[(Xi, x, Xj)] -= 1
                if support_counter[(Xi, x, Xj)] == 0:
                    csp.prune(Xi, x, removals)
                    revised = True
                    unsupported_variable_value_pairs.append((Xi, x))
            if revised:
                if not csp.curr_domains[Xi]:
                    return False, checks  # CSP is inconsistent
    return True, checks  # CSP is satisfiable


# ______________________________________________________________________________
# CSP Backtracking Search

# Variable ordering 

def first_unassigned_variable(assignment, csp):
    """The default variable order."""
    return first([var for var in csp.variables if var not in assignment])

def mrv(assignment, csp):
    """Minimum-remaining-values heuristic."""
    return argmin_random_tie([v for v in csp.variables if v not in assignment],
                             key=lambda var: num_legal_values(csp, var, assignment))

def num_legal_values(csp, var, assignment):
    if csp.curr_domains:
        return len(csp.curr_domains[var])
    else:
        return count(csp.nconflicts(var, val, assignment) == 0 for val in csp.domains[var])

# Value ordering 

def unordered_domain_values(var, assignment, csp):
    """The default value order."""
    return csp.choices(var)

def lcv(var, assignment, csp):
    """Least-constraining-values heuristic."""
    return sorted(csp.choices(var), key=lambda val: csp.nconflicts(var, val, assignment))

# Inference

def no_inference(csp, var, value, assignment, removals):
    return True

def forward_checking(csp, var, value, assignment, removals):
    """Prune neighbor values inconsistent with var=value."""
    for B in csp.neighbors[var]:
        if B not in assignment:
            for b in csp.curr_domains[B][:]:
                global counter 
                counter+=1 #This is the counter of the measurements
                if not csp.constraints(var, value, B, b):
                    if var not in csp.conflicts[B]:
                        csp.conflicts[B].append(var) #adding var to the conflict set of B because it caused incosistency
                    csp.prune(B, b, removals)
            if not csp.curr_domains[B]:
                for con in csp.conflicts[B]:
                    if con not in csp.conflicts[var] and con!=var:
                        csp.conflicts[var].append(con) #var occured domain annihilation to B so we add the conflict set of B to var
                return False
    return True

def mac(csp, var, value, assignment, removals, constraint_propagation=AC3b):
    """Maintain arc consistency."""
    return constraint_propagation(csp, {(X, var) for X in csp.neighbors[var]}, removals)

# The search, proper 

def backtracking_search(csp, select_unassigned_variable=first_unassigned_variable,
                        order_domain_values=unordered_domain_values, inference=no_inference):
    """[Figure 6.5]"""
    def backtrack(assignment):
        if len(assignment) == len(csp.variables):
            return assignment
        var = select_unassigned_variable(assignment, csp)
        for value in order_domain_values(var, assignment, csp):
            global assignmCounter
            assignmCounter+=1
            if 0 == csp.nconflicts(var, value, assignment):
                csp.assign(var, value, assignment)
                removals = csp.suppose(var, value)
                if inference(csp, var, value, assignment, removals):
                    if counter > counterLimit: #This is the limit of the method measurements
                        return -1
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                csp.restore(removals)
        csp.unassign(var, assignment)
        return None

    result = backtrack({})
    assert result is None or result == -1 or csp.goal_test(result)
    return result

# ______________________________________________________________________________
# Min-conflicts Hill Climbing search for CSPs

def min_conflicts(csp, max_steps=100000):
    """Solve a CSP by stochastic Hill Climbing on the number of conflicts."""
    # Generate a complete assignment for all variables (probably with conflicts)
    csp.current = current = {}
    for var in csp.variables:
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    # Now repeatedly choose a random conflicted variable and change it
    for i in range(max_steps):
        conflicted = csp.conflicted_vars(current)
        if not conflicted:
            return current
        var = random.choice(conflicted)
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    return None

def min_conflicts_value(csp, var, current):
    """Return the value that will give var the least number of conflicts.
    If there is a tie, choose at random."""
    return argmin_random_tie(csp.domains[var], key=lambda val: csp.nconflicts(var, val, current))

# ______________________________________________________________________________

def make_arc_consistent(Xj, Xk, csp):
    """Make arc between parent (Xj) and child (Xk) consistent under the csp's constraints,
    by removing the possible values of Xj that cause inconsistencies."""
    # csp.curr_domains[Xj] = []
    for val1 in csp.domains[Xj]:
        keep = False  # Keep or remove val1
        for val2 in csp.domains[Xk]:
            if csp.constraints(Xj, val1, Xk, val2):
                # Found a consistent assignment for val1, keep it
                keep = True
                break

        if not keep:
            # Remove val1
            csp.prune(Xj, val1, None)

    return csp.curr_domains[Xj]

def assign_value(Xj, Xk, csp, assignment):
    """Assign a value to Xk given Xj's (Xk's parent) assignment.
    Return the first value that satisfies the constraints."""
    parent_assignment = assignment[Xj]
    for val in csp.curr_domains[Xk]:
        if csp.constraints(Xj, parent_assignment, Xk, val):
            return val

    # No consistent assignment available
    return None 

#implementing dom_wdeg algorithm for weight-based variable ordering
def dom_wdeg(assignment,csp):
    min=999999
    minVar=0
    for var in csp.variables:
        sum=1
        if var not in assignment:
            #here I calculate the weight of the var and it's neighbor's that have not yet been assigned so as to find their sum (all)
            for neighbor in csp.neighbors[var]:
                if neighbor not in assignment:
                    temp=csp.constr_dict[str(var)+' '+str(neighbor)][1]
                    temp=int(temp)
                    #the sum is here
                    sum=sum+temp
            #here I choose the variable with the minimum current_domain_size/weight_count (variable : sum)
            if len(csp.curr_domains[var])/sum < min:
                min=len(csp.curr_domains[var])/sum 
                minVar=var
    return minVar

#This is implemented like the backtracking algorithm above but with adjustments so we can backtrack more than 1 nodes
def fc_cbj(csp, select_unassigned_variable=first_unassigned_variable,
           order_domain_values=unordered_domain_values, inference=no_inference):

    def backjump(assignment):
        if csp.goal_test(assignment):  # Termination state
            print("FC-CBJ solution:", assignment)
            print("Constraint checks = ", counter)
            return None

        variable = select_unassigned_variable(assignment, csp)

        for value in order_domain_values(variable, assignment, csp):
            global assignmCounter
            assignmCounter += 1

            if csp.nconflicts(variable, value, assignment) == 0:
                csp.assign(variable, value, assignment)
                removals = csp.suppose(variable, value)

                if inference(csp, variable, value, assignment, removals):
                    if counter > counterLimit:
                        return -1

                    result = backjump(assignment)

                    if result == -1:
                        return -1
                    if result is None:
                        return None
                    if result == -1000:
                        csp.restore(removals)
                        continue
                    if result != variable:
                        csp.unassign(variable, assignment)
                        for i in range(len(csp.conflicts)):
                            if variable in csp.conflicts[i]:
                                csp.conflicts[i].remove(variable)
                        csp.restore(removals)
                        return result

                csp.restore(removals)

        csp.unassign(variable, assignment)

        for i in range(len(csp.conflicts)):
            if variable in csp.conflicts[i]:
                csp.conflicts[i].remove(variable)

        if not csp.conflicts[variable]:
            return -1000

        conf = csp.conflicts[variable][-1]

        for con in csp.conflicts[variable]:
            if con not in csp.conflicts[conf] and con != conf:
                csp.conflicts[conf].append(con)

        return conf

    result = backjump({})
    return result


class RLFA(CSP):
    def __init__(self,f1,f2,f3):
        variables=[]
        domains={}
        neighbors={}
        #Dictionary  that holds each constraint for all variables as we do  not have a constant constraint for all variables
        self.constr_dict={} 
        self.conflicts={} #This is used by the fc-cbj algorithm
        flag=0
        #here we set up the domains temporary list
        domains={}
        for row in f2:
            #skip 1st row it does not have information to be used in the assignment
            #we use variable flag to symbolize if a line is read or not 
            if flag==0:
                flag=1
                continue
            tempList=[]
            i=2
            line=row[i]
            digit="empty"
            #read the whole line and put numbers in a list
            while line!='\n':
                while not line.isspace() and line!='\n':
                    if digit!="empty":
                        digit+=line
                    else:
                        digit=line
                    i+=1
                    line=row[i]
                digit=int(digit)
                tempList.append(digit)
                digit="empty"
                if line!='\n':
                    i+=1
                    line=row[i]
            #this is the domains list based on dom.txt files
            domains[int(row[0])]=tempList
        #Assign every variable to it's domain list (ex. in var2-f24 the variable 0 has the 0 list of domains assigned to it)
        flag=0
        for ch in f1:
            #skip 1st row
            if flag==0:
                flag=1
                continue
            domainList="empty"
            i=0
            sp=ch[0]
            while not sp.isspace():
                if domainList!="empty":
                    domainList=domainList+sp
                else:
                    domainList=sp
                i+=1
                sp=ch[i]
            domainList=int(domainList)  
            variables.append(domainList)
            self.conflicts[domainList]=[]
            #setting up the domains for each variable based on its index in the dom file
            i+=1
            domains[domainList]=domains[int(ch[i])].copy()
        #setting up neighbours here
        flag=0
        for row in f3:
            if flag==0:
                for ch in range(0,len(variables)):
                    neighbors[ch]=[]
                flag=1
                continue
            #variables firstNum and secondNum are set to -inf before the procedure of assigning neighbours
            firstNum=-99999
            secNum=-99999
            x=row[0]
            domainList="empty"
            i=0
            while not x.isspace():
                if domainList!="empty":
                    domainList=domainList+x
                else:
                    domainList=x
                i+=1
                x=row[i]
            domainList=int(domainList)  
            firstNum=domainList
            domainList="empty"
            i+=1
            x=row[i]
            while not x.isspace():
                if domainList!="empty":
                    domainList=domainList+x
                else:
                    domainList=x
                i+=1
                x=row[i]
            domainList=int(domainList)  
            secNum=domainList
            #Assign a  neighbor for each of the 2 variables
            neighbors[firstNum].append(secNum)
            neighbors[secNum].append(firstNum)
            #Adding the constraint beween the 2 variables to our dictionary
            i+=1
            x=row[i]
            domainList="empty"
            while x!='\n':
                if domainList!="empty":
                    domainList=domainList+x
                else:
                    domainList=x
                i+=1
                x=row[i]
            self.constr_dict[str(firstNum)+' '+str(secNum)]=(domainList,1) #this is the weight of the dom/wdeg heuristic
            self.constr_dict[str(secNum)+' '+str(firstNum)]=(domainList,1) #by doing this we add for both variables the neighbour
        super().__init__(variables, domains, neighbors, self.constraintFunction)

    def constraintFunction(self,A,a,B,b):
        #Implemented this function to : a) trun strings to integers and b) calculate 2 variables based on their constraints
        constraint=self.constr_dict[str(A)+' '+str(B)]
        operator=constraint[0][0]
        k=constraint[0][2]
        for i in range(3,len(constraint[0])):
            k+=constraint[0][i]
        k=int(k)
        if operator=='>':
            if abs(a-b) > k:
                return True
            else:
                if(len(self.curr_domains[B])==1):
                    #Strategy to aboid domain wipeout for varibale B by increasing the weight used by the wdeg algorithm
                    temp=self.constr_dict[str(A)+' '+str(B)]
                    temp2=temp[1]
                    temp2=int(temp2)
                    temp2+=1
                    self.constr_dict[str(A)+' '+str(B)]=(temp[0],temp2)
                    self.constr_dict[str(B)+' '+str(A)]=(temp[0],temp2)
                return False
        if operator=='=':
            if abs(a-b) == k:
                return True
            else:
                if(len(self.curr_domains[B])==1):
                    #Same strategy as above 
                    temp=self.constr_dict[str(A)+' '+str(B)]
                    temp2=temp[1]
                    temp2=int(temp2)
                    temp2+=1
                    self.constr_dict[str(A)+' '+str(B)]=(temp[0],temp2)
                    self.constr_dict[str(B)+' '+str(A)]=(temp[0],temp2)
                return False


    
if __name__ == "__main__":
    #the 3 files 1 of each category to be used in the radio link frequency assignment 
 
    f1 = open("../RLFA Dataset/var2-f24.txt", "r")
    f2 = open("../RLFA Dataset/dom2-f24.txt", "r")
    f3 = open("../RLFA Dataset/ctr2-f24.txt", "r")

    solution=RLFA(f1,f2,f3)
    startTime = time.time()

    #fc algorithm
    counterLimit=40000000
    run=backtracking_search(solution, select_unassigned_variable=dom_wdeg, inference=forward_checking)
    if run==None:
        print("FC: There is no solution!!\n","Constraint checks =",counter)
    elif run==-1:
        print("FC: ","Constraint checks >",counter)
    else:
        print("FC: ",run,"\nConstraint checks =",counter)

    print("CPU time:",time.time()-startTime)
    if (assignmCounter != 0):
        print("Assignments:",assignmCounter,"\n")

    startTime = time.time()

    #mac algorithm
    counterLimit=1000000
    run=backtracking_search(solution, select_unassigned_variable=mrv, inference=mac)
    if run==None:
        print("MAC: There is no solution!!\n",counter)
    elif run==-1:
        print("MAC: ","Constraint checks >",counter)
    else:
        print("MAC: ",run,"\nConstraint checks =",counter)

    print("CPU time:",time.time()-startTime)
    if (assignmCounter != 0):
        print("Assignments:",assignmCounter,"\n")

    startTime = time.time()


    #fc-cbj algorithm
    counterLimit=40000000
    run=fc_cbj(solution, select_unassigned_variable=dom_wdeg, inference=forward_checking)
    if run==-1:
        print("FC-CBJ: ","Constraint checks >",counter)
    elif run!=None:
        print("FC-CBJ: There is no solution!!","\nConstraint checks =",counter)



    print("CPU time:",time.time()-startTime)
    if (assignmCounter != 0):
        print("Assignments:",assignmCounter,"\n")
    #when done close the files     
    f1.close()
    f2.close()
    f3.close()
