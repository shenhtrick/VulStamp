import pandas as pd
import networkx as nx
import re


def parse_line(location):
    if pd.isna(location) or '\\' in str(location):
        return None
    try:
        return int(str(location).split(':')[0])
    except (ValueError, IndexError):
        return None


def is_control_flow(edge_type):
    return edge_type == 'CONTROLS'


nodes = pd.read_csv("nodes.csv", sep='\t')
edges = pd.read_csv("edges.csv", sep='\t')

nodes['line'] = nodes['location'].apply(parse_line)
valid_nodes = nodes.dropna(subset=['line']).copy()
valid_nodes['line'] = valid_nodes['line'].astype(int)

function_entry_lines = valid_nodes[valid_nodes['type'] == 'Function'].set_index('key')['line'].to_dict()

node_line_map = {}
for _, row in nodes.iterrows():
    if row['type'] in ['CFGEntryNode', 'CFGExitNode']:
        func_id = row['functionId']
        node_line_map[row['key']] = function_entry_lines.get(func_id, None)
    else:
        node_line_map[row['key']] = parse_line(row['location'])

edges['start_line'] = edges['start'].map(node_line_map)
edges['end_line'] = edges['end'].map(node_line_map)
edges = edges.dropna(subset=['start_line', 'end_line']).astype({'start_line': int, 'end_line': int})

control_flow_graph = nx.DiGraph()
for _, row in edges[edges['type'] == 'CONTROLS'].iterrows():
    src, dst = row['start_line'], row['end_line']
    if src != dst:
        control_flow_graph.add_edge(src, dst)

transitive_closure = nx.transitive_closure(control_flow_graph)

line_control_deps = {}
for src, dst in transitive_closure.edges:
    if src == dst:
        continue
    line_control_deps.setdefault(src, set()).add(dst)

line_data_deps = {}
for _, row in edges[edges['type'] == 'REACHES'].iterrows():
    src, dst, var = row['start_line'], row['end_line'], row.get('var', '')
    if src != dst and var:
        line_data_deps.setdefault(src, {}).setdefault(var, set()).add(dst)

call_deps = {}
pattern = re.compile(r'(\w+)\s*\(([^)]*)\)')

for _, row in valid_nodes.iterrows():
    code = row['code']
    line = row['line']
    if pd.notna(code):
        # 查找函数调用
        matches = pattern.findall(code)
        for match in matches:
            func_name = match[0]
            # 查找被调用函数的入口行号
            callee_node = valid_nodes[
                (valid_nodes['type'] == 'Function') & (valid_nodes['code'].str.contains(func_name))]
            if not callee_node.empty:
                callee_line = callee_node['line'].iloc[0]
                call_deps.setdefault(line, set()).add(callee_line)

return_deps = {}
return_pattern = re.compile(r'return\s+([^;]+);')

for func_id, entry_line in function_entry_lines.items():

    func_lines = valid_nodes[valid_nodes['functionId'] == func_id]['line'].sort_values().tolist()
    if not func_lines:
        continue
    last_line = func_lines[-1]

    return_line = None
    for line in func_lines:
        code = valid_nodes[valid_nodes['line'] == line]['code'].values[0]
        if pd.notna(code) and 'return' in code:
            match = return_pattern.search(code)
            if match:
                return_line = line
                break

    if return_line is None:
        return_line = last_line

    for caller_line in call_deps:
        if entry_line in call_deps.get(caller_line, set()):
            return_deps.setdefault(return_line, set()).add(caller_line)

line_code_map = valid_nodes.groupby('line')['code'].apply(lambda x: '\n'.join(x.dropna().unique())).to_dict()

for node_key, line in node_line_map.items():
    if line is not None and line not in line_code_map:
        node = nodes[nodes['key'] == node_key]
        if not node.empty:
            code = node['code'].values[0]
            if pd.notna(code):
                line_code_map[line] = code

def build_pdg():
    pdg = nx.DiGraph()

    for line in line_code_map:
        pdg.add_node(line, code=line_code_map[line])

    for src in line_control_deps:
        for dst in line_control_deps[src]:
            pdg.add_edge(src, dst, type='CONTROL')

    for src in line_data_deps:
        for var, targets in line_data_deps[src].items():
            for dst in targets:
                pdg.add_edge(src, dst, type='DATA', var=var)

    for src in call_deps:
        for dst in call_deps[src]:
            pdg.add_edge(src, dst, type='CALL')

    for src in return_deps:
        for dst in return_deps[src]:
            pdg.add_edge(src, dst, type='RETURN')

    return pdg

pdg = build_pdg()

# 打印 PDG 结构
print("Program Dependence Graph (PDG):")
print(f"Nodes: {pdg.number_of_nodes()}")
print(f"Edges: {pdg.number_of_edges()}")
print("\nEdges with Attributes:")
for src, dst, attrs in pdg.edges(data=True):
    edge_type = attrs.get('type', 'UNKNOWN')
    var = attrs.get('var', '')
    print(f"Line {src} → Line {dst} [{edge_type}] {f'({var})' if var else ''}")

with open('sensiAPI.txt', 'r') as f:
    system_apis = [line.strip() for line in f if line.strip()]

user_defined_functions = valid_nodes[valid_nodes['type'] == 'Function']['code'].dropna().unique()

def find_interesting_lines(code_map, system_apis, user_defined_functions):
    interesting_lines = []
    call_pattern = re.compile(r'(\w+)\s*\(([^)]*)\)')
    operator_pattern = re.compile(r'(?<!\*)[\+\-\=\*\/](?!\*)')

    for line, code in code_map.items():
        if pd.notna(code):
            # 查找运算符
            if operator_pattern.search(code):
                interesting_lines.append(line)
            # 查找函数调用
            matches = call_pattern.findall(code)
            for match in matches:
                func_name = match[0]
                if func_name in system_apis or func_name in user_defined_functions:
                    interesting_lines.append(line)
                    break
    return interesting_lines

interesting_lines = find_interesting_lines(line_code_map, system_apis, user_defined_functions)

print("\nInterest Points:")
for line in interesting_lines:
    code = line_code_map[line]
    print(f"Line {line}: {code}")

def forward_slice(pdg, start_line):
    visited = set()
    queue = [start_line]
    visited.add(start_line)
    while queue:
        current = queue.pop(0)
        for predecessor in pdg.predecessors(current):
            # 只考虑数据依赖
            edge = pdg.get_edge_data(predecessor, current)
            if edge and edge.get('type') == 'DATA':
                if predecessor not in visited:
                    visited.add(predecessor)
                    queue.append(predecessor)
    return visited
def backward_slice(pdg, start_line):
    visited = set()
    queue = [start_line]
    visited.add(start_line)
    while queue:
        current = queue.pop(0)
        for successor in pdg.successors(current):
            if successor not in visited:
                visited.add(successor)
                queue.append(successor)
    return visited

all_slice_lines = set()
for line in interesting_lines:
    forward = forward_slice(pdg, line)
    backward = backward_slice(pdg, line)
    all_slice_lines.update(forward)
    all_slice_lines.update(backward)

sliced_pdg = pdg.subgraph(all_slice_lines)

print("\nSliced Program Dependence Graph (PDG):")
print(f"Nodes: {sliced_pdg.number_of_nodes()}")
print(f"Edges: {sliced_pdg.number_of_edges()}")
print("\nEdges with Attributes:")
for src, dst, attrs in sliced_pdg.edges(data=True):
    edge_type = attrs.get('type', 'UNKNOWN')
    var = attrs.get('var', '')
    print(f"Line {src} → Line {dst} [{edge_type}] {f'({var})' if var else ''}")

print("\nDependencies:")
sorted_slice_lines = sorted(all_slice_lines)
for line in sorted_slice_lines:
    code = line_code_map[line]
    print(f"Line {line}: {code}")

    if line in line_control_deps:
        sorted_control_deps = sorted(line_control_deps[line])
        print(f"  - Control Dependencies → Lines: {sorted_control_deps}")

    if line in line_data_deps:
        for var, targets in line_data_deps[line].items():
            sorted_data_deps = sorted(targets)
            print(f"  - Data Dependency [{var}] → Lines: {sorted_data_deps}")

    if line in call_deps:
        sorted_call_deps = sorted(call_deps[line])
        print(f"  - Call Dependencies → Lines: {sorted_call_deps}")

    if line in return_deps:
        sorted_return_deps = sorted(return_deps[line])
        print(f"  - Return Value → Lines: {sorted_return_deps}")

    print("-" * 80)