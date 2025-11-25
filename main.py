import json
import sys
import re
import os
from pycparser import c_parser, c_ast, c_generator
from BufferTrace import run_trace_driven_pipeline

CONTEXT_LINES = 5
parser = c_parser.CParser()
generator = c_generator.CGenerator()

macro_cache = {}

def remap_path(path, path_remap):
    parts = path.split(os.sep)
    for i, p in enumerate(parts):
        if p == path_remap:
            return os.sep.join(parts[i:])
    return path

def load_macros(file_path, path_remap):
    file_path = remap_path(file_path, path_remap)
    if file_path in macro_cache:
        return macro_cache[file_path]

    macros = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#define'):
                    parts = line.split(None, 2)
                    if len(parts) >= 3:
                        name = parts[1]
                        if '(' not in name:
                            value = parts[2].split('//')[0].strip()
                            macros[name] = value
                    elif len(parts) == 2:
                        macros[parts[1]] = ''
    except:
        pass
    macro_cache[file_path] = macros
    return macros

def expand_macros(code_line, file_path, path_remap):
    macros = load_macros(file_path, path_remap)
    if not macros:
        return code_line

    result = code_line
    for name in sorted(macros.keys(), key=len, reverse=True):
        value = macros[name]
        result = re.sub(rf'\b{re.escape(name)}\b', value, result)
    return result

def get_line(file_path, line_num, path_remap, expand=False):
    real_path = remap_path(file_path, path_remap)
    try:
        with open(real_path, 'r') as f:
            lines= f.readlines()
        if 0 < line_num <= len(lines):
            line = lines[line_num - 1].strip()
            if expand:
                line = expand_macros(line, file_path, path_remap)
            return line
    except:
        pass
    return None

def get_source_context(file_path, line_num, path_remap, context=CONTEXT_LINES):
    file_path = remap_path(file_path, path_remap)
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        start = max(0, line_num - context - 1)
        end = min(len(lines), line_num + context)
        return {i+1: lines[i].rstrip() for i in range(start, end)
        }
    except FileNotFoundError:
        return f'[File not found: {file_path}]'
    except Exception as e:
        return f'[Error reading file: {e}]'

def wrap_statement(code):
    return f'void __f__() {{ {code} }}'

def node_to_code(node):
    return generator.visit(node)

def get_array_name(node):
    if isinstance(node.name, c_ast.ArrayRef):
        return get_array_name(node.name)
    return node_to_code(node.name)

def analyze_condition(cond):
    info = {}
    if isinstance(cond, c_ast.BinaryOp):
        info['operator'] = cond.op
        info['left'] = node_to_code(cond.left)
        info['right'] = node_to_code(cond.right)
        if cond.op in ('<', '>', '<=', '>='):
            code = node_to_code(cond)
            if any(x in code.lower() for x in ['size', 'len', 'count', 'max', 'num', 'cap']):
                info['is_bounds_check'] = True
                info['bounds_check_type'] = 'size_comparison'

        if cond.op in ('==', '!=') and node_to_code(cond.right) in ('NULL', '0', 'nullptr'):
            info['is_null_check'] = True
    if isinstance(cond, c_ast.UnaryOp):
        if cond.op == '!':
            info['is_negation'] = True
            info['operand'] = node_to_code(cond.expr)
    return info

def analyze_bounds_check(file_path, bug_line):
    file_path = remap_path(file_path, path_remap)
    checks = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        start = max(0, bug_line - 10)
        for i in range(start, bug_line):
            line = lines[i].strip()
            classification = classify_line(line, file_path, bug_line)
            if classification.get('type') == 'if_statement':
                cond_info = classification.get('condition_info', {})
                if cond_info.get('is_bounds_check') or cond_info.get('is_null_check'):
                    checks.append({
                        'line': i+1,
                        'code': line,
                        'classification': classification
                    })
    except:
        pass
    return checks

def analyze_node(node):
    result = {'ast_type': type(node).__name__}

    if isinstance(node, c_ast.If):
        result['type'] = 'if_statement'
        result['condition'] = node_to_code(node.cond)
        result['condition_info'] = analyze_condition(node.cond)
    elif isinstance(node, c_ast.While):
        result['type'] = 'while_loop'
        result['condition'] = node_to_code(node.cond)
    elif isinstance(node, c_ast.For):
        result['type'] = 'for_loop'
        if node.cond:
            result['condition'] = node_to_code(node.cond)
    elif isinstance(node, c_ast.Return):
        result['type'] = 'return'
        if node.expr:
            result['value'] = node_to_code(node.expr)
    elif isinstance(node, c_ast.Assignment):
        result['type'] = 'assignment'
        result['lvalue'] = node_to_code(node.lvalue)
        result['rvalue'] = node_to_code(node.rvalue)
        if isinstance(node.lvalue, c_ast.ArrayRef):
            result['is_array_write'] = True
            result['array_name'] = get_array_name(node.lvalue)
            result['array_index'] = node_to_code(node.lvalue.subscript)
    elif isinstance(node, c_ast.FuncCall):
        result['type'] = 'function_call'
        result['function'] = node_to_code(node.name)
        if node.args:
            result['arguments'] = [node_to_code(a) for a in node.args.exprs]
        fname = result['function']
        if fname in ('strcpy', 'strcat', 'sprintf', 'gets', 'scanf'):
            result['potentially_unsafe'] = True
        if fname in ('malloc', 'calloc', 'realloc', 'free'):
            result['memory_op'] = fname
    elif isinstance(node, c_ast.Decl):
        result['type'] = 'declaration'
        result['name'] = node.name
        if isinstance(node.type, c_ast.ArrayDecl) and node.type.dim:
            result['is_array'] = True
            result['array_size'] = node_to_code(node.type.dim)
    else:
        result['type'] = 'other'
    return result

def handle_multiline(file_path, line_num):
    real_path = remap_path(file_path, path_remap)
    try:
        with open(real_path, 'r') as f:
            lines = f.readlines()

        if line_num <= 0 or line_num > len(lines):
            return None

        statement = lines[line_num - 1].strip()
        incomplete_patterns = [
            (r'if\s*\(.*\)\s*{?\s*$', 'if_block'),
            (r'else\s+if\s*\(.*\)\s*{?\s*$', 'if_block'),
            (r'else\s*{?\s*$', 'else_block'),
            (r'while\s*\(.*\)\s*{?\s*$', 'while_block'),
            (r'for\s*\(.*\)\s*{?\s*$', 'for_block'),
            (r'.*\|\|\s*$', 'continuation'),
            (r'.*&&\s*$', 'continuation'),
            (r'.*,\s*$', 'continuation')
        ]

        is_incomplete = False
        for pattern, _ in incomplete_patterns:
            if re.match(pattern, statment):
                is_incomplete = True
                break

        if not is_incomplete:
            return statement

        if re.match(r'(if|while|for|else\s+if)\s*\(.*\)\s*$', statement):
            return statement + " { int __dummy__; }"

        if re.match(r'(if|while|for|else\s+if)\s*\(.*\)\s*{\s*$', statement):
            return statement + " }"

        if re.match(r'else\s*{\s*$', statement):
            return statement + " }"

        if re.search(r'(\|\||&&|,)\s*$', statement):
            complete = statement
            for i in range(line_num, min(line_num + 5, len(lines))):
                next_line = lines[i].strip()
                complete += " " + next_line
                if next_line.endswith(')') or next_line.endswith(';'):
                    if not re.search(r'{', complete):
                        complete += " { int __dummy__; }"
                    return complete
        return statement
    except:
        return None

def parse_line(code_line, file_path=None, line_num=None):
    if not code_line or not code_line.strip():
        return {'type': 'empty'}

    try:
        ast = parser.parse(wrap_statement(code_line))
        body = ast.ext[0].body.block_items
        if body:
            return analyze_node(body[0])
    except Exception as e:
        if file_path and line_num:
            complete = handle_multiline(file_path, line_num)
            if complete and complete != code_line:
                try:
                    ast = parser.parse(wrap_statment(complete))
                    func_body = ast.ext[0].body.block_items
                    if func_body:
                        result = analyze_node(func_body[0])
                        result['multiline'] = True
                        result['complete_code'] = complete
                        return result
                except:
                    pass
        return {
            'type': 'parse_error',
            'error': str(e),
            'raw': code_line
        }
    return {'type': 'unknown', 'raw': code_line} 

def find_array_accesses(code_line):
    accesses = []
    try:
        ast = parser.parse(wrap_statement(code_line))
        stack = [ast]
        while stack:
            node = stack.pop()
            if isinstance(node, c_ast.ArrayRef):
                accesses.append({
                    'array': get_array_name(node),
                    'index': node_to_code(node.subscript)
                })
            for _, child, in node.children():
                stack.append(child)
    except:
        pass
    return accesses

def classify_line(line, path_remap, file_path=None, line_num=None):
    if not line:
        return {'type': 'empty'}
    expanded = line
    if file_path:
        expanded = expand_macros(line, file_path, path_remap)

    result = {'raw': line}
    if expanded != line:
        result['expanded'] = expanded

    parsed = parse_line(expanded, file_path, line_num)
    result.update(parsed)

    if parsed.get('type') not in ['empty', 'parse_error']:
        accesses = find_array_accesses(expanded)
        if accesses:
            result['array_accesses'] = accesses
    return result

def simplify_steps(steps, results, bug_func, bug_file, path_remap):
    seen = set()
    simplified = {}
    last_seen_func = ''
    last_seen_file = ''
    for step in steps:
        real_path = remap_path(step['File'], path_remap)
        line = step['Line']
        tip = step['Tip']
        if not tip:
            tip = 'Calling trace related func'
        if step['Line'] <= 0:
            continue

        for r in results:
            if r['name'] == step['FuncName']:
                line -= r['disjunction']
                break
        key = (line, tip, real_path)
        if key in seen:
            continue
        seen.add(key)

        line_code = get_line(real_path, line, path_remap)
        while line_code and not line_code.endswith((';', '{', '}', '(', ')')):
            line += 1
            line_code += get_line(real_path, line, path_remap)
        if line not in simplified:
            simplified[line] = {}
            if last_seen_file == '' or real_path != last_seen_file:
                last_seen_file = real_path
                simplified[line]['Cur_FileName'] = real_path
            if last_seen_func == '' or step['FuncName'] != last_seen_func:
                last_seen_func = step['FuncName']
                simplified[line]['Cur_FuncName'] = step['FuncName']
            simplified[line]['code'] = line_code
            simplified[line]['tips'] = []
        simplified[line]['tips'].append(tip)
    for code_info in simplified.values():
        tips = code_info['tips']
        branch_tips = [t for t in tips if t.startswith('Branching')]
        if len(branch_tips) > 1:
            non_branch = [t for t in tips if not t.startswith('Branching')]
            tips[:] = non_branch + ['Branching (multiple)']
    return simplified

def find_bug_location(steps):
    for step in reversed(steps):
        if 'overrun' in step.get('Tip', '').lower():
            return step
    return steps[-1] if steps else None

def process_report(report, results, path_remap):
    steps = report.get('DiagSteps', [])
    if not steps:
        return None

    bug_step = find_bug_location(steps)
    if not bug_step:
        return None

    file_path = remap_path(bug_step['File'], path_remap)
    bug_line = bug_step['Line']
    func_name = bug_step['FuncName']

    bug_code = get_line(file_path, bug_line, path_remap)

    source_context = get_source_context(file_path, bug_line, path_remap)
    simple_steps = simplify_steps(steps, results, func_name, file_path, path_remap)

    return {
        'File': remap_path(file_path, path_remap),
        'FuncName': func_name,
        'BugLine': bug_line,
#        'bug_code': bug_code,
#         'bug_classification': bug_classification,
#        'nearby_bounds_checks': nearby_checks,
        'SourceContext': source_context,
        'DiagSteps': simple_steps
    }

def main():
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <src-dir> <report.json>')
        sys.exit(1)
    print()
    print('=' * 10 + ' CodV2 ' + '=' * 10)
    print('\nBeginning report generation...\n')

    src_dir = sys.argv[1]
    report_path = sys.argv[2]
    results = run_trace_driven_pipeline(src_dir, report_path)

    with open(report_path, 'r') as f:
        data = json.load(f)

    processed = [process_report(r, results, src_dir) for r in data.get('Reports', [])]
    
    output = {
            'TotalBugs': data.get('TotalBugs', len(data.get('Reports', []))),
            'Reports': processed
    }

    result = json.dumps(output, indent=2)
    out_file = f'final_report_{src_dir}_v2.json'
    with open(out_file, 'w') as f:
        f.write(result)
        print(f'Wrote report to {out_file}.')

if __name__ == '__main__':
    main()
