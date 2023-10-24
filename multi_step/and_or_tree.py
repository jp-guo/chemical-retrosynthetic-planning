class And_Or_node:
    def __init__(self, smiles='', mode=0) -> None:
        self.smiles = smiles
        # mode = 0: molecole
        # mode = 1: template
        self.mode = mode
        self.done = False
        self.father = None
        self.children = []
        self.best_route = False
        if mode == 0:
            self.cost = 0
        
    def add_child(self, child):
        self.children.append(child)
        
    def set_father(self, father):
        self.father = father
    
    def evaluate_upward_propagation(self):
        # molecole
        if self.mode == 0:
            if len(self.children) != 0 and any([child.done for child in self.children]):
                self.done = True
                
        if self.mode == 1:
            if len(self.children) != 0 and all([child.done for child in self.children]):
                self.done = True
                
        if self.done == True and self.father is None:
            self.best_route = True
            return True
        
        
        # print(self.smiles,self.done)
        
        if self.done == True and self.father != None:
            res = self.father.evaluate_upward_propagation()
            if res:
                self.best_route = True
            return res
        
        return False
    
def find_best_route(root):
    ...
            
        

            
if __name__ == '__main__':
    root = And_Or_node(mode=0,smiles='root')
    ##############################
    r1 = And_Or_node(mode=1,smiles='r1')
    root.add_child(r1)
    r1.set_father(root)
    
    r2 = And_Or_node(mode=1,smiles='r2')
    root.add_child(r2)
    r2.set_father(root)
    ################################
    m1 = And_Or_node(mode=0,smiles='m1')
    r1.add_child(m1)
    m1.set_father(r1)
    
    m2 = And_Or_node(mode=0,smiles='m2')
    r1.add_child(m2)
    m2.set_father(r1)
    
    m3 = And_Or_node(mode=0,smiles='m3')
    r1.add_child(m3)
    m3.set_father(r1)
    
    ##################################
    m1.done = True
    res = m1.evaluate_upward_propagation()
    print(res)
    m2.done = False
    
    m3.done = True
    res = m3.evaluate_upward_propagation()
    print(res)
    ####################################
    m4 = And_Or_node(mode=0,smiles='m4')
    r2.add_child(m4)
    m4.set_father(r2)
    
    m5 = And_Or_node(mode=0,smiles='m5')
    r2.add_child(m5)
    m5.set_father(r2)
    ###################################
    m4.done = True
    res = m4.evaluate_upward_propagation()
    print(res)
    
    m5.done = True
    res = m5.evaluate_upward_propagation()
    print(res)
    
    
    



        
        
        