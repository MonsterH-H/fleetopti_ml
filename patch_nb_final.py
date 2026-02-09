import json
import os
import glob
import re

notebook_dir = r'c:\mon-apprentissage-dev\fleetopti-ml\notebooks'
notebooks = glob.glob(os.path.join(notebook_dir, "*.ipynb"))

def patch_source(source):
    new_source = []
    for line in source:
        # Handle load_*_data unpacking
        if re.search(r'load_(?:co2|maintenance|logistics)_data\(', line):
            if "=" in line and ", _ =" not in line and ", encoders =" not in line:
                line = re.sub(r'(\w+)\s*=\s*(load_(?:co2|maintenance|logistics)_data)', r'\1, _ = \2', line)
        
        # Handle prepare_splits unpacking
        if "prepare_splits(" in line:
            if "scaler =" in line and "feature_names" not in line and ", _ =" not in line:
                line = line.replace("scaler =", "scaler, _ =")
        
        # Robust corr(numeric_only=True)
        if ".corr(" in line:
            if "numeric_only" not in line:
                if ".corr()" in line:
                    line = line.replace(".corr()", ".corr(numeric_only=True)")
                else:
                    line = line.replace(".corr(", ".corr(numeric_only=True, ")
            
        new_source.append(line)
    return new_source

for nb_path in notebooks:
    print(f"Patching {nb_path}...")
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        changed = False
        first_code_cell = True
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                old_source = cell.get('source', [])
                
                # Ensure autoreload is at the top of the first encountered code cell
                if first_code_cell:
                    full_text = "".join(old_source)
                    if "%load_ext autoreload" not in full_text:
                        old_source.insert(0, "%load_ext autoreload\n")
                        old_source.insert(1, "%autoreload 2\n")
                        changed = True
                    first_code_cell = False
                
                new_source = patch_source(old_source)
                if new_source != old_source:
                    cell['source'] = new_source
                    changed = True
        
        if changed:
            with open(nb_path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=4)
            print(f"  Fixed {nb_path}")
        else:
            print(f"  No changes needed for {nb_path}")
    except Exception as e:
        print(f"  Error: {e}")

print("Done.")
