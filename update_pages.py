import os
import re

# Get all page files
pages_dir = "pages"
page_files = [f for f in os.listdir(pages_dir) if f.endswith(".py") and f != "0_Authentication.py"]

# Authentication check code to add
auth_check_code = """
# Add authentication check
from utils.auth_redirect import check_authentication
check_authentication()
"""

# Process each page file
for page_file in page_files:
    file_path = os.path.join(pages_dir, page_file)
    
    # Read the file content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Find the position to insert the authentication check
    # We want to insert it after the imports and page config, but before the main content
    # Look for the last import or the page_config section
    import_pattern = r'(from .+ import .+|import .+)'
    imports = re.findall(import_pattern, content)
    
    if imports:
        last_import = imports[-1]
        insert_pos = content.find(last_import) + len(last_import)
        
        # Find the next non-empty line after the last import
        next_line_pos = content.find('\n', insert_pos)
        while next_line_pos < len(content) - 1 and content[next_line_pos+1] == '\n':
            next_line_pos = content.find('\n', next_line_pos + 1)
        
        insert_pos = next_line_pos + 1
    else:
        # If no imports found, insert at the beginning
        insert_pos = 0
    
    # Insert the authentication check
    new_content = content[:insert_pos] + auth_check_code + content[insert_pos:]
    
    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.write(new_content)
    
    print(f"Updated {page_file}")

print("All pages updated to require authentication.")
