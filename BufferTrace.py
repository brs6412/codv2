import os
import re
import json
from collections import defaultdict


def load_targets_from_trace(trace_json_path: str):
    """
    Build targets: {basename.c: {func_name: earliest_line}} from the trace.
    If multiple steps reference the same function, we keep the earliest line number.
    """
    with open(trace_json_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    targets = defaultdict(dict)  # basename -> {func_name: earliest_line}
    branch_lines = defaultdict(dict)
    seen_funcs = set()
    for bug in report.get("Reports", []):
        for step in bug.get("DiagSteps", []):
            file_path = step.get("File", "")
            func = step.get("FuncName", "")
            if func not in seen_funcs:
                seen_funcs.add(func)
            else:
                continue
            line = step.get("Line", None)
            tip = step.get("Tip", "")
            if "branch" in tip and not branch_lines[func]:
                branch_lines[func] = line

            # Extract trailing file basename
            base = os.path.basename(file_path)
            if not (base.endswith(".c") or base.endswith(".h")):
                m = re.search(r'([^/\\]+\.(ch))$', file_path or "")
                if m:
                    base = m.group(1)
                else:
                    continue

            if not func:
                continue

            # Track earliest line per (file, func)
            if line is None:
                # If no line provided, default to 1 (so closest-to-top wins)
                line = 1
            prev = targets[base].get(func)
            targets[base][func] = min(prev, line) if prev is not None else line

    return targets, branch_lines


def find_best_definition_line(lines, func_name, target_line):
    """
    Return the 1-based line index that contains func_name and is closest to target_line.
    If multiple matches tie, choose the earliest occurrence.
    """
    occurrences = []
    # Simple substring search; optionally tighten to word-boundary
    # pattern = re.compile(r'\b' + re.escape(func_name) + r'\b')
    for idx, line in enumerate(lines, start=1):
        if func_name in line:
            occurrences.append(idx)

    if not occurrences:
        return None

    # Choose the occurrence with minimal absolute distance to target_line
    best = min(occurrences, key=lambda i: (abs(i - target_line), i))
    return best


def accumulate_body_from(lines, start_line):
    """
    Given a start_line (1-based), accumulate a function body using brace matching.
    We start counting from the first '{' on or after start_line.
    """
    # Find the first '{' at or after start_line
    i = start_line
    while i <= len(lines) and '{' not in lines[i - 1]:
        # Stop early if we encounter a semicolon-only prototype before any '{'
        if ';' in lines[i - 1] and '{' not in lines[i - 1]:
            return start_line, start_line, ""  # treat as no body
        i += 1

    if i > len(lines):
        return start_line, start_line, ""  # no opening brace found

    brace_count = 0
    body_lines = []
    j = i
    while j <= len(lines):
        brace_count += lines[j - 1].count('{')
        brace_count -= lines[j - 1].count('}')
        body_lines.append(lines[j - 1])
        if brace_count == 0 and j >= i:
            break
        j += 1

    end_line = j if j <= len(lines) else len(lines)
    body_text = "".join(body_lines)
    return i, end_line, body_text


def analyze_file(path, func_to_targetline):
    """
    For each target function in this file, pick the best definition line and accumulate body.
    """
    results = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for func_name, target_line in func_to_targetline.items():
        best_line = find_best_definition_line(lines, func_name, target_line)
        if best_line is None:
            results.append({
                "file": path,
                "name": func_name,
                "start_line": None,
                "end_line": None,
                "body": "",
                "note": "function name not found in file"
            })
            continue

        brace_start, end_line, body = accumulate_body_from(lines, best_line)
        header = lines[best_line-1:brace_start-1]
        header = "".join(header)
        results.append({
            "file": path,
            "name": func_name,
            "func_header": header,
            "start_line": brace_start,  # where the '{' is
            "end_line": end_line,
            "body": body
        })
    return results


def run_trace_driven_pipeline(project_root: str, trace_json_path: str):
    """
    Resolve basenames to paths in project_root, then analyze only trace-listed functions.
    If multiple files share the same basename, pick the first found.
    """
    targets, branches = load_targets_from_trace(trace_json_path)

    all_results = []
    # Build a simple index from project_root: basename -> path
    index = {}
    for root, _, files in os.walk(project_root):
        for f in files:
            if f.endswith(".c") and f not in index:
                index[f] = os.path.join(root, f)

    for base, funcmap in targets.items():
        path = index.get(base)
        if not path:
            print(f"Warning: no file found for {base}")
            continue
        results = analyze_file(path, funcmap)
        all_results.extend(results)

    for func_result in all_results:
        if func_result["name"] in branches.keys():
            compute_and_store_disjunction(func_result, branches.get(func_result["name"]))
        else:
            print("Warning: no branch statements in function body, no disjunction calculation")
            func_result["disjunction"] = 0
    return all_results


def compute_and_store_disjunction(func_result, trace_branch_line):
    """
    Compute the first branch disjunction for a function and store only the offset number.
    """
    body_lines = func_result["body"].splitlines()
    start_line = func_result["start_line"]

    branch_pattern = re.compile(r'\b(if|else if|else|switch|for|while|do|assert)\b')

    code_branch_line = None
    for offset, line in enumerate(body_lines, start=0):
        stripped = line.strip()
        if stripped.startswith("/*") or stripped.startswith("//") or stripped.startswith("#"):
            continue
        if branch_pattern.search(stripped):
            code_branch_line = start_line + offset
            break

    if code_branch_line is None:
        func_result["disjunction"] = None
        return func_result

    # Store only the offset (trace - code)
    func_result["disjunction"] = trace_branch_line - code_branch_line
    return func_result


def index_into_body_with_disjunction(func_result, trace_line):
    """
    Given a trace line, use the stored disjunction offset to index into the function body.
    Instead of returning a single line, accumulate the full logical statement:
      - End at ';' for declarations/statements
      - End at '{' for control structures (loops, if, switch)
    """
    offset = func_result.get("disjunction")
    if offset is None:
        return None, None

    corrected_line = trace_line - offset

    # Clamp to function bounds
    corrected_line = max(func_result["start_line"], min(corrected_line, func_result["end_line"]))

    body_lines = func_result["body"].splitlines()
    idx = corrected_line - func_result["start_line"]
    if not (0 <= idx < len(body_lines)):
        return corrected_line, None

    # Start accumulating from corrected_line
    stmt_lines = []
    for j in range(idx, len(body_lines)):
        line = body_lines[j]
        # Remove leading tabs/spaces only
        normalized = line.lstrip(" \t").rstrip()
        stmt_lines.append(normalized)

        stripped = normalized.strip()
        if stripped.endswith(";") or stripped.endswith("{"):
            break

    full_stmt = " ".join(stmt_lines).strip()
    return corrected_line, full_stmt


def is_type_token(tok: str) -> bool:
    """
    Heuristic: treat a token as a type if it contains a known type substring.
    This catches typedefs like uint16_t, int32_t, etc.
    """
    base_types = ["int", "char", "float", "double", "long", "short", "struct", "NULL", "enum", "union", "const"]
    return any(bt in tok for bt in base_types)


def tokenize(line: str):
    """
    Split a line into tokens: identifiers, numbers, and punctuation.
    Whitespace, string literals, and char literals are ignored.
    """
    # Remove string literals
    no_strings = re.sub(r'"([^"\\]|\\.)*"', ' ', line)
    # Remove char literals
    no_chars = re.sub(r"'([^'\\]|\\.)*'", ' ', no_strings)
    # Remove single line comments
    no_single_comms = re.sub(r"\*.*?\*", ' ', no_chars)

    # Identifiers, numbers, and punctuation (no spaces)
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S", no_single_comms)


def collect_vars_from_statement(stmt_text: str, found_vars: []):
    # Keyword sets
    c_keywords = {
        "if", "else", "switch", "case", "for", "while", "do", "return", "goto",
        "break", "continue", "sizeof", "assert"
    }

    tokens = tokenize(stmt_text)
    vars_found = {"declared": [], "used": []}

    # Extra check for strange typedefs and vars
    if (tokens[0] not in c_keywords and not is_type_token(tokens[0])
            and tokens[0] not in found_vars and len(tokens) >= 2
            and re.match(r"[A-Za-z_][A-Za-z0-9_]*", tokens[0])):
        # If line ends with ';' and looks like a declaration
        if ";" in tokens:
            # Skip first token (type), collect subsequent identifiers
            for t in tokens[1:]:
                # print(t)
                # print(t)
                if re.match(r"[A-Za-z_][A-Za-z0-9_]*", t) and t not in c_keywords:
                    vars_found["declared"].append(t)
            return vars_found

    if is_type_token(tokens[0]):
        # Declaration: collect following identifiers until ; or {
        j = 1

        # Skip next token if type is struct or enum
        if tokens[0] == ("struct" or "enum" or "union" or "const"):
            j += 1
        while j < len(tokens):
            if is_type_token(tokens[j]) or tokens[j] in c_keywords:
                break
            if re.match(r"[A-Za-z_][A-Za-z0-9_]*", tokens[j]):
                vars_found["declared"].append(tokens[j])
            if tokens[j] in [";", "{"]:
                break
            j += 1
    else:
        for i in range(0, len(tokens)):
            # Usage heuristic: identifier not keyword, not type, not function call
            if (re.match(r"[A-Za-z_][A-Za-z0-9_]*", tokens[i])
                    and tokens[i] not in c_keywords
                    and not is_type_token(tokens[i])):
                # skip function calls and jump labels
                if i + 1 < len(tokens) and (tokens[i + 1] == "(" or tokens[i + 1] == ":"):
                    continue

                # skip goto labels
                if i != 0 and tokens[i-1] == "goto":
                    continue

                vars_found["used"].append(tokens[i])
    return vars_found


def build_var_dictionary(func_result):
    """
    Merge declared + used variables across all statements in a function body.
    """
    body_lines = func_result["body"].splitlines()
    start_line = func_result["start_line"]

    var_dict = {}
    found_vars = []

    # Accumulate logical statements (end at ; or {)
    buffer = []
    stmt_start = None
    for offset, line in enumerate(body_lines, start=0):
        stripped = line.strip()
        if not stripped or stripped.startswith(("/*", "*", "*//", "//", "#")):
            continue

        if stmt_start is None:
            stmt_start = start_line + offset

        # Remove leading indentation but keep internal spaces
        buffer.append(line.lstrip(" \t").rstrip())

        if stripped.endswith(";") or stripped.endswith("{") or stripped.endswith("}"):
            stmt_text = " ".join(buffer).strip()
            vars_found = collect_vars_from_statement(stmt_text, found_vars)

            for v in vars_found["declared"]:
                var_dict.setdefault(v, {"declared": True, "used": False})
            for v in vars_found["used"]:
                entry = var_dict.setdefault(v, {"declared": False, "used": False})
                entry["used"] = True

            found_vars = list(var_dict.keys())

            buffer = []
            stmt_start = None

    return var_dict


# --- Example usage ---
if __name__ == "__main__":
    project_root = "scan_source_code/tcpdump"  # adjust as needed
    trace_json_path = "init_traces/final_report_tcpdump.json"
    results = run_trace_driven_pipeline(project_root, trace_json_path)

    for r in results:
        name = r["name"]
        start = r["start_line"]
        end = r["end_line"]
        print(f"{r['file']}: {name} lines {start}–{end}")
        print(r.get("func_header"))
        print(r.get("note", ""), end="")
        print(r["body"])
        print(r["disjunction"])
        print("-" * 60)

    print()

    print(results[0]["disjunction"])
    print(build_var_dictionary(results[2]))

    # # Test using offset to get correct code line from trace
    # trace_line = 1062
    # print(results[2]["disjunction"])
    # corrected_line, code_text = index_into_body_with_disjunction(results[2], trace_line)
    # print(f"Trace line {trace_line} corrected to {corrected_line}: {code_text}")
