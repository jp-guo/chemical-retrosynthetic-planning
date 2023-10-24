from and_or_tree import And_Or_node
from models import one_step_model, evaluation_model
from dataset import get_dataset
import heapq
from heap_node import HeapNode


def main():
    B_model = one_step_model()
    E_model = evaluation_model()
    
    
    starting_mols, target_mol_route, target_mols = get_dataset()
    
    with open('result.csv', 'w') as f:
        for target_mol in target_mols:
            route = retrosynthetic_planning(B_model, E_model, starting_mols, target_mol)
            if route is None:
                f.write(f'{target_mol},False\n')
            else:
                f.write(f'{target_mol},True,{route}')
        
def retrosynthetic_planning(B_model, E_model, starting_mols, target_mol):
    root = And_Or_node(smiles=target_mol, mode=0)
    heap:list[HeapNode] = []
    heapq.heappush(heap, HeapNode(root, E_model(root.smiles)))
    while True:
        # no molecules left before finding a route
        if len(heap) == 0:
            break
        
        mol:And_Or_node = heapq.heappop(heap).data
        templates = B_model(mol.smiles)
        
        # fail to expand
        if templates is None:
            continue
        
        for template in templates:
            t_node = And_Or_node(template,1)
            mol.add_child(t_node)
            t_node.set_father(mol)
            
            reactants = get_reactants(template, mol)
            
            for reactant in reactants:
                m_node = And_Or_node(reactant, 0)
                t_node.add_child(m_node)
                m_node.set_father(t_node)
            
            for m_node in t_node.children:
                if m_node in starting_mols:
                    m_node.evaluate_upward_propagation()
                    if root.done:
                        best_route = get_best_route(root)
                        return best_route
                else:
                    heapq.heappush(heap, HeapNode(m_node, E_model(m_node.smiles)))
    return None
                
def get_reactants(template, target_mol):
    # TODO: finish this part.
    ...
    
def get_best_route(root):
    # TODO: finish this part
    ...
    
if __name__ == '__main__':
    main()