import pickle
import gzip
import time
import matplotlib.pyplot as plt
import networkx as nx

def get_dataset():
    starting_mols_root = r'../data/Multi-Step task/starting_mols.pkl.gz'
    target_mol_route_root = r'../data/Multi-Step task/target_mol_route.pkl'
    test_mols_root = r'../data/Multi-Step task/test_mols.pkl'

    s1 = time.time()
    # starting_mols = pickle.load(gzip.open(starting_mols_root, "rb") ,encoding='latin1')
    # starting_mols = list(starting_mols)
    e1 = time.time()

    s2 = time.time()
    with open(target_mol_route_root,'rb') as f:
        target_mol_route = pickle.load(f)
    e2 = time.time()

    s3 = time.time()
    with open(test_mols_root, 'rb') as f:
        test_mols = pickle.load(f)
    e3 = time.time()
    
    print(f'Time of loading:starting mol:{e1-s1:.2f} target mol:{e2-s2:.2f} test mol:{e3-s3:.2f}')
    print(len(target_mol_route), len(test_mols))
    
    for r in target_mol_route[1]:
        print(r)
    print('-'*20)
    print(test_mols[1])
    
    exit()
    mols = set()
    routes = []
    for sub_routes in target_mol_route:
        routes += sub_routes
        
    for reaction in routes:
        a,b = reaction.split('>>')
        mols.add(a)
        mols.add(b)
        
    print('=='*10)
    # print(list(mols))
    mols = list(mols)
    # for mol in mols:
    #     print(mol)
        
    mol_dict = {}
    for i, mol in enumerate(mols):
        mol_dict[mol] = i
    print('=='*10)
    
    for reaction in routes:
        a,b = reaction.split('>>')
        print(mol_dict[a],'>>',mol_dict[b])
    
        
    # for tar in target_mols:
    #     print(len(tar))
    exit()
    print(target_mols[0])
    print(len(target_mols[0]))
    print(len(target_mols))
    
    print(test_mols[0])
    print(len(test_mols))
    
    
def test_task1_dataset():
    
    mode = ['train', 'test', 'val'][0]

    with open(f'./data/schneider50k/raw_{mode}.csv','r') as f:
        lines = f.readlines()
    lines = lines[1:]
    reactions = [line.strip().split(',')[2] for line in lines]
    # reactions = [line.split('>>') for line in lines]
    for reaction in reactions:
        reactant, product = reaction.split('>>')
        # if product.find('.') != -1:
        #     print(reaction)
        #     break
        if reactant.find('.') != -1:
            print(reaction)
            break
    
def graph_vis():
    target_mol_route_root = r'./data/Multi-Step task/target_mol_route.pkl'
    with open(target_mol_route_root,'rb') as f:
        target_mol_route = pickle.load(f)
        
    mols = set()
    routes = []
    for sub_routes in target_mol_route:
        routes += sub_routes
    
    split_routes = []
    for route in routes:
        reaction, product = route.split('>>')
        if '.' in product:
            for p in product.split('.'):
                split_routes.append(f'{reaction}>>{p}')
            
    # routes = routes[:10]
    
    for reaction in split_routes:
        a,b = reaction.split('>>')
        a = a.split('.')
        for item in a:
            mols.add(item)
        # print(reaction)
        # mols.add(a)
        mols.add(b)
    
    mol_dict = {}
    for i, mol in enumerate(mols):
        mol_dict[mol] = i
        
    G = nx.DiGraph()
    for i in range(len(mols)):
        G.add_node(i, desc=f'v{i}')
    
    for reaction in split_routes:
        a,b = reaction.split('>>')
        G.add_edge(mol_dict[a],mol_dict[b])
    
    # nx.draw_networkx(G, with_labels=None)
    # 画出标签
    # node_labels = nx.get_node_attributes(G, 'desc')
    # nx.draw_networkx_labels(G,  labels=node_labels)
    # 画出边权值
    # edge_labels = nx.get_edge_attributes(G, 'name')
    # nx.draw_networkx_edge_labels(G,  edge_labels=edge_labels)
    nx.draw(G, 
        # with_labels=True, #这个选项让节点有名称
        edge_color='b', # b stands for blue! 
        # pos=nx.circle_layout(G), # 这个是选项选择点的排列方式，具体可以用 help(nx.drawing.layout) 查看
     # 主要有spring_layout  (default), random_layout, circle_layout, shell_layout   
     # 这里是环形排布，还有随机排列等其他方式  
        # pos=nx.spring_layout(G),
        # pos=nx.random_layout(G),
        pos=nx.shell_layout(G),
        node_color='r', # r = red
        node_size=10, # 节点大小
        width=0.1, # 边的宽度
       )


    plt.title('AOE_CPM', fontsize=10)
    plt.show()
    
        
if __name__ == '__main__':
    print('=='*20+'begin'+'=='*20)
    get_dataset()
    # graph_vis()
    # test_task1_dataset()
    