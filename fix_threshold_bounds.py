#!/usr/bin/env python3
import re

# Read the current file
with open('src/training/model_editing.py', 'r') as f:
    content = f.read()

# Find and replace the problematic section
old_pattern = r'''        if threshold_idx >= len\(sorted_scores\):
            threshold_idx = len\(sorted_scores\) - 1
        
        threshold_value = sorted_scores\[threshold_idx\]\.item\(\)'''

new_pattern = '''        # Check if we have any scores to work with
        if len(sorted_scores) == 0:
            # No active parameters available - return infinity to skip pruning
            return float('inf')
        
        # Ensure threshold_idx is valid
        if threshold_idx < 0:
            threshold_idx = 0
        elif threshold_idx >= len(sorted_scores):
            threshold_idx = len(sorted_scores) - 1
        
        threshold_value = sorted_scores[threshold_idx].item()'''

# Replace the pattern
content = re.sub(old_pattern, new_pattern, content, flags=re.MULTILINE)

# Write back the fixed file
with open('src/training/model_editing.py', 'w') as f:
    f.write(content)

print("✅ Fixed threshold bounds checking!")
