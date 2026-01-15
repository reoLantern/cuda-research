import os
import re
import sys

def get_files(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # We focus on .cu files as per instruction, but user said "all files" in the folder.
            # However, user also said "you only need to look at .cu files".
            # I will stick to matching any file but maybe prioritize text files?
            # Let's just look at all files, but catch read errors.
            if file.endswith('.cu'):
                file_paths.append(os.path.join(root, file))
    return file_paths

def extract_instructions(file_paths, prefix):
    # Regex: \bPREFIX[\w.]*
    # We escape prefix just in case, though usually it's plain text.
    pattern = re.compile(r'\b' + re.escape(prefix) + r'[\w.]*')
    
    found = set()
    
    for fp in file_paths:
        try:
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Determine if we should only look in comments.
                # User's files have SASS in comments.
                # Scanning full file content is easier and unlikely to produce false positives 
                # if the prefix is specific (like F2FP).
                # However, to be cleaner, we can scan line by line and look for //
                
                # Let's stick to scanning the whole content as it's more robust against 
                # weird formatting, and "matches to space" is the requirement.
                # But matching "F2FP" inside "AF2FP" is bad. \b handles start.
                
                matches = pattern.findall(content)
                for m in matches:
                    # Filter out the prefix itself if it appears bare? 
                    # "F2FP" is a valid instruction.
                    # Ensure it's a SASS instruction. SASS usually (user example) inside comments.
                    # But if we just grep, we might catch C++ variables.
                    # Given the user context "sass compiled ... as comments", I will try to match only lines with //
                    
                    found.update(matches)
        except Exception as e:
            print(f"Error reading {fp}: {e}")
            
    return sorted(list(found))

def tokenize(instruction):
    return instruction.split('.')

def simplify_tokens(token_lists):
    """
    Takes a list of list-of-tokens.
    Returns a string representation with {} and ().
    """
    if not token_lists:
        return ""
    
    # 1. Common Prefix (Optimization & formatting)
    common_prefix = []
    while token_lists and all(l and l[0] == token_lists[0][0] for l in token_lists):
        token = token_lists[0][0]
        common_prefix.append(token)
        token_lists = [l[1:] for l in token_lists]
    
    prefix_str = ".".join(common_prefix)
    
    # If done
    if not token_lists:
        return prefix_str
        
    # 2. Separate Empty (Termination) from Non-Empty
    empty_lists = [l for l in token_lists if not l]
    non_empty_lists = [l for l in token_lists if l]
    
    has_empty = len(empty_lists) > 0
    
    if not non_empty_lists:
        return prefix_str

    # 3. Group by Suffix (using last token)
    groups = {}
    for l in non_empty_lists:
        last = l[-1]
        if last not in groups:
            groups[last] = []
        groups[last].append(l[:-1])
        
    group_results = []
    
    for last_token, sub_lists in groups.items():
        # Recursively simplify the middle part
        middle = simplify_tokens(sub_lists)
        
        # Assemble middle + last
        if not middle:
            # consumed everything
            full = last_token
        elif middle.startswith('{'):
             full = middle + "." + last_token
        elif middle.startswith('.'):
             full = middle + "." + last_token
        else:
             full = middle + "." + last_token
             
        group_results.append(full)
        
    group_results.sort()
    
    # 4. Final assembly
    if len(group_results) == 1:
        combined = group_results[0]
    else:
        combined = "(" + "|".join(group_results) + ")"
        
    if has_empty:
        # If combined starts with '.', strip it for inside brace usually?
        # But here combined is built from token strings.
        # Format: {combined}
        # Avoid double dots ..
        
        # If combined is `(A|B)`. -> `{(A|B)}` or `{.A|.B}`?
        # Let's just wrap result. `{.combined}`.
        if combined.startswith('{'):
             combined = "{." + combined + "}"
        else:
             combined = "{." + combined + "}"
             
    # Prepend prefix
    if prefix_str:
        if combined.startswith('{'):
            return prefix_str + combined
        else:
            return prefix_str + "." + combined
    else:
        return combined

def run_analysis(directory, prefix):
    print(f"Processing directory: {directory}")
    files = get_files(directory)
    instructions = extract_instructions(files, prefix)
    
    print(f"Found {len(instructions)} unique instructions starting with '{prefix}':")
    for ins in instructions:
        print(ins)
        
    print("-" * 20)
    print("Summary Pattern:")
    print("-" * 20)
    
    if not instructions:
        print("No instructions found.")
        print("-" * 20)
        return

    # To generate summary, we assume the prefix is the instruction base and stripped usually?
    # Actually, tokenize everything.
    # Instruction strings include the prefix.
    # e.g. "F2FP.F16", "F2FP.F32"
    
    token_lists = [tokenize(ins) for ins in instructions]
    summary = simplify_tokens(token_lists)
    
    # Cleaning up the summary for display
    # My logic might produce `F2FP.{.SAT}.mid` -> `F2FP{.SAT}.mid`
    # Let's see results.
    print(summary)
    print("-" * 20)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        run_analysis(sys.argv[1], sys.argv[2])
    else:
        print("Starting interactive mode. Press Ctrl+C to exit.")
        try:
            while True:
                d = input("Enter directory (default: current): ").strip()
                if not d:
                    d = os.getcwd()
                elif not os.path.exists(d):
                    print(f"Directory {d} does not exist.")
                    continue
                
                p = input("Enter instruction prefix (e.g. F2FP): ").strip()
                if not p:
                    print("Prefix cannot be empty.")
                    continue
                    
                run_analysis(d, p)
                print("\n" + "="*40 + "\n")
        except KeyboardInterrupt:
            print("\nExiting.")
